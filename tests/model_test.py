from theia import RetrievalModel
import tensorflow as tf


model = RetrievalModel()
# model.fit()

user = "42"
print("Recommendations for user {}: {}".format(
    user, model.recommend(user)[0, :3]))
