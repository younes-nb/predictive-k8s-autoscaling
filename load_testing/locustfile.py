import math
import random
import time
from locust import FastHttpUser, TaskSet, LoadTestShape
from faker import Faker

fake = Faker()
products = [
    "0PUK6V6EV0",
    "1YMWWN1N4O",
    "2ZYFJ3GM2N",
    "66VCHSJNUP",
    "6E92ZMYYFZ",
    "9SIQT8TOJO",
    "L9ECAV7KIM",
    "LS4PSXUNUM",
    "OLJCESPC7Z",
]


class UserBehavior(TaskSet):
    def on_start(self):
        self.client.get("/")

    tasks = {
        lambda l: l.client.get("/"): 1,
        lambda l: l.client.post(
            "/setCurrency", {"currency_code": random.choice(["EUR", "USD", "JPY"])}
        ): 2,
        lambda l: l.client.get("/product/" + random.choice(products)): 10,
        lambda l: l.client.get("/cart"): 3,
    }


class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]

    def wait_time(self):
        return random.expovariate(1.0 / 3.0)


class StatisticalLoadShape(LoadTestShape):
    warmup_duration = 300
    warmup_users = 20

    avg_users = 100
    macro_amplitude = 60
    macro_cycle = 1200

    micro_amplitude = 40
    micro_cycle = 180

    noise_factor = 0.35
    time_limit = 3900

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        if run_time < self.warmup_duration:
            return (self.warmup_users, max(1, int(self.warmup_users / 5)))

        test_time = run_time - self.warmup_duration

        macro_trend = self.avg_users + (
            self.macro_amplitude * math.sin(2 * math.pi * test_time / self.macro_cycle)
        )

        micro_trend = self.micro_amplitude * math.sin(
            2 * math.pi * test_time / self.micro_cycle
        )

        base_tick_users = max(10, macro_trend + micro_trend)

        std_dev = math.sqrt(base_tick_users) * self.noise_factor
        stochastic_users = int(random.gauss(base_tick_users, std_dev))

        final_user_count = max(1, stochastic_users)

        spawn_rate = max(1, int(final_user_count / 3))

        return (final_user_count, spawn_rate)
