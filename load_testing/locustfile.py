import math
import random
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
        return random.expovariate(1.0 / 4.0)


class StatisticalLoadShape(LoadTestShape):
    warmup_duration = 3600
    test_duration = 10800
    time_limit = warmup_duration + test_duration
    warmup_users = 50
    avg_users = 250
    macro_amplitude = 120
    macro_cycle = 3600
    micro_amplitude = 60
    micro_cycle = 300
    noise_factor = 0.50

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        if run_time < self.warmup_duration:
            current_warmup = int((run_time / self.warmup_duration) * self.warmup_users)
            return (max(1, current_warmup), 2)

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

        current_minute = int(test_time / 60)
        is_burst_window = (current_minute % 15 == 0) or (current_minute % 15 == 1)

        if is_burst_window:
            burst_multiplier = random.uniform(1.5, 2.0)
            stochastic_users = int(stochastic_users * burst_multiplier)

        if random.random() < 0.02:
            stochastic_users = int(stochastic_users * 0.7)

        final_user_count = max(1, stochastic_users)

        spawn_rate = max(5, int(final_user_count / 2))

        return (final_user_count, spawn_rate)
