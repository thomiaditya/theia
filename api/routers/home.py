from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import HTMLResponse

router = APIRouter(
    tags=["home"]
)

@router.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Theia</title>
        </head>
        <body>
            <h1>Theia</h1>
            <p> This is a recommendation engine for mental health routers.</p>
            <p> For more information, visit <a href="/docs">Our Swagger Documentation</a>.</p>
        </body>
    </html>
    """