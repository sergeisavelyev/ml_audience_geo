import json
import pandas as pd


class YandexAudienceLoader:

    def __init__(self, path):
        self.path = path

    def load(self):

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def to_dataframe(self):

        data = self.load()

        rows = []

        for segment_id, payload in data.items():

            gender = payload.get("gender", {})

            age = payload.get("age", {})

            row = {
                "segment_id": int(segment_id),
                "totals": payload.get("totals", 0),

                "male_share": gender.get("1", 0),
                "female_share": gender.get("0", 0),

                "age_18_24": age.get("18", 0),
                "age_25_34": age.get("25", 0),
                "age_35_44": age.get("35", 0),
                "age_45_54": age.get("45", 0),
                "age_55_plus": age.get("55", 0),

                "self_similarity": payload.get("self_similarity", 0),
                "ios_share": payload.get("device", {}).get("4", 0),
            }

            rows.append(row)

        return pd.DataFrame(rows)