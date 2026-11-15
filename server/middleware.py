from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # FIXME: replace with your frontend URL
        allow_methods=["*"],
        allow_headers=["*"],
    )
