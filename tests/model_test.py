from theia import RetrievalModel

model = RetrievalModel()
# model.train()
print(model.recommend("42", "last_saved"))
# model.save()
