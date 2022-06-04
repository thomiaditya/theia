from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import HTMLResponse
from theia import RetrievalModel


router = APIRouter(
    prefix="/recommender",
    tags=["recommender"]
)

def train(epochs=1):
    print("Training the model...")
    model = RetrievalModel(epochs=epochs)
    model.train()
    print("Model trained with {} epochs".format(epochs))


@router.get("/recommend/{user_id}")
async def recommend(user_id: str):
    result = RetrievalModel.static_recommend(str(user_id))

    # Change the result of tf Tensor to a list
    result = result.numpy().tolist()[0]

    return {"status": "success", "result": result}


@router.get("/train/{epochs}")
async def train_model(epochs: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(train, epochs)
    return {"status": "success", "message": "Training started"}