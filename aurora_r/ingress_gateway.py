"""Optional HTTP/1.x experiment ingress instrumenter and bounded reverse proxy.

For gRPC/WebSocket/HTTP2/streaming traffic use equivalent instrumentation at
its actual ingress. This reference gateway buffers request/response bodies and
is NOT a general production ingress replacement. Configure traffic to pass
through it or the independent SLO counters will be absent and scaling holds.
"""
import argparse,asyncio,os,time

def eligible_path(path, include_prefixes=(), exclude_prefixes=()):
    """Only configured user-facing routes enter the SLO cohort.

    Application semantics (e.g. whether invalid checkout inputs are legitimate
    SLO requests) are not inferable from HTTP status codes. Operators must
    identify eligible paths before the request begins.
    """
    if any(path.startswith(prefix) for prefix in exclude_prefixes):return False
    return not include_prefixes or any(path.startswith(prefix) for prefix in include_prefixes)
from .request_lifecycle import RequestObserver,classify_failure

def main(argv=None):
    from aiohttp import web,ClientSession,ClientTimeout
    from prometheus_client import generate_latest,CONTENT_TYPE_LATEST
    ap=argparse.ArgumentParser()
    ap.add_argument('--upstream',required=True)
    ap.add_argument('--namespace',required=True)
    ap.add_argument('--deadline-ms',type=float,required=True)
    ap.add_argument('--port',type=int,default=8090)
    ap.add_argument('--max-lifetime-s',type=float,default=60)
    ap.add_argument('--max-body-bytes',type=int,default=1024*1024)
    ap.add_argument('--operation',default='default')
    ap.add_argument('--eligible-path-prefix',action='append',default=[],
                    help='User-facing request paths counted in the SLO; repeatable')
    ap.add_argument('--exclude-path-prefix',action='append',default=[],
                    help='Excluded paths such as probes, static content and intentionally invalid operations')
    args=ap.parse_args(argv)
    if args.deadline_ms/1000>=args.max_lifetime_s:raise ValueError('max-lifetime must exceed SLO deadline')
    observer=RequestObserver(args.deadline_ms/1000,args.namespace)
    app=web.Application(client_max_size=args.max_body_bytes)
    upstream=args.upstream.rstrip('/')
    state={'session':None,'watchdog':None}
    async def start(app):
        observer.ensure_operation(args.operation)
        observer.sweep()  # publish a heartbeat even before the first request
        state['session']=ClientSession(timeout=ClientTimeout(total=args.max_lifetime_s))
        async def sweep():
            while True:
                observer.sweep();await asyncio.sleep(min(.05,args.deadline_ms/2000))
        state['watchdog']=asyncio.create_task(sweep())
    async def stop(app):
        state['watchdog'].cancel()
        try:await state['watchdog']
        except asyncio.CancelledError:pass
        await state['session'].close()
    app.on_startup.append(start);app.on_cleanup.append(stop)
    async def metrics(request):
        return web.Response(body=generate_latest(),content_type=CONTENT_TYPE_LATEST.split(';')[0])
    async def proxy(request):
        # A fixed operation name avoids cardinality explosions from unique URLs.
        # Health and /metrics are intentionally NOT eligible user operations.
        operation=args.operation
        eligible=eligible_path(request.path,args.eligible_path_prefix,args.exclude_path_prefix)
        token=observer.start(operation) if eligible else None
        cause='unknown'
        try:
            body=await request.read()
            if len(body)>args.max_body_bytes:
                cause='business';return web.Response(status=413,text='request too large')
            banned={'host','connection','transfer-encoding','content-length','accept-encoding',
                    'x-aurora-failure-cause','x-envoy-response-flags','x-application-rejection'}
            headers={k:v for k,v in request.headers.items() if k.lower() not in banned}
            url=upstream+request.rel_url.path_qs
            async with state['session'].request(request.method,url,headers=headers,data=body,allow_redirects=False) as rsp:
                response_body=await rsp.read()
                cause=classify_failure(rsp.status,rsp.headers)
                allowed={'content-type','cache-control','location','x-request-id'}
                rh={k:v for k,v in rsp.headers.items() if k.lower() in allowed}
                return web.Response(status=rsp.status,body=response_body,headers=rh)
        except asyncio.TimeoutError:
            # A timeout is a deadline violation, but NOT automatically capacity.
            return web.Response(status=504,text='upstream timeout')
        except Exception:
            return web.Response(status=502,text='upstream unavailable')
        finally:
            if token is not None:observer.finish(token,cause)
    app.router.add_get('/metrics',metrics)
    app.router.add_get('/healthz',lambda request:web.Response(text='ok'))
    app.router.add_route('*','/{tail:.*}',proxy)
    web.run_app(app,port=args.port)
if __name__=='__main__':main()