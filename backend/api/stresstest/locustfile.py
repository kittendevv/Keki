from locust import HttpUser, between, task


class KekiUser(HttpUser):
    wait_time = between(1, 3)  # wait 1-3 seconds between requests

    @task(3)
    def get_recipe(self):
        self.client.get("/recipe/pizza")

    @task(3)
    def get_similar(self):
        self.client.get("/similar/pizza")

    @task(2)
    def get_nutrition(self):
        self.client.get("/nutrition/pizza")

    @task(2)
    def get_portion(self):
        self.client.get("/portion/pizza")

    @task(1)
    def health_check(self):
        self.client.get("/health")
