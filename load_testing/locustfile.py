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
    avg_users = 100
    amplitude = 80
    cycle_length = 3600
    noise_factor = 0.1
    time_limit = 3600

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        base_tick_users = self.avg_users + (
            self.amplitude * math.sin(2 * math.pi * run_time / self.cycle_length)
        )

        std_dev = math.sqrt(max(1, base_tick_users)) * self.noise_factor
        stochastic_users = int(random.gauss(base_tick_users, std_dev))

        final_user_count = max(1, stochastic_users)
        spawn_rate = max(1, int(final_user_count / 10))

        return (final_user_count, spawn_rate)
