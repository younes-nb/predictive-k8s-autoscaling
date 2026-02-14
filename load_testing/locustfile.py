import random
from locust import FastHttpUser, TaskSet, between, LoadTestShape
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
    wait_time = between(1, 5)


class DoubleSpikeShape(LoadTestShape):
    stages = [
        {"duration": 120, "users": 20, "spawn_rate": 1},
        {"duration": 300, "users": 20, "spawn_rate": 1},
        {
            "duration": 360,
            "users": 150,
            "spawn_rate": 20,
        },
        {"duration": 480, "users": 150, "spawn_rate": 1},
        {"duration": 600, "users": 10, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data
        return None
