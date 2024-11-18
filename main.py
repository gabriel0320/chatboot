from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from chatbot import Chatbot
import uvicorn

# Instancia del chatbot
chatbot = Chatbot('intents.json')

# Crear la aplicación FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    texto_usuario = data['text']
    respuesta = chatbot.obtener_respuesta(texto_usuario)
    return JSONResponse(content={"response": respuesta})

@app.get("/")
def get_html():
    with open("static/index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
