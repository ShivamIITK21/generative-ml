import base64
import requests

class Visualizer:
    def __init__(self):
        self.add_url = "http://localhost:8080/add"
        self.remove_url = "http://localhost:8080/remove"

    @staticmethod
    def toB64(s: str):
        return base64.b64encode(s.encode("ascii"))

    def draw(self, id: str, exp: str):
        requests.get(url =self.add_url, params={"id": self.toB64(id), "exp": self.toB64(exp)})

    def remove(self, id: str):
        requests.get(url=self.remove_url, params={"id": self.toB64(id)})

    @staticmethod
    def getLatexKV(key: float, val: float) -> str:
        return r"""\{0 \le y <""" + f"""{val}""" r"""\}"""+ r"""x=""" + f"""{key}"""


