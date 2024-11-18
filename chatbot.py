import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import random
import unicodedata
import pickle

nltk.download('punkt')

class Chatbot:
    def __init__(self, intents_file):
        self.intents = self.load_intents(intents_file)
        self.classifier = self.train_classifier()

    def normalizar_texto(self, texto):
        texto = texto.lower()
        texto = ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        )
        return texto

    def load_intents(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)

    def preparar_datos(self, intents):
        entrenamiento = []
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                words = word_tokenize(self.normalizar_texto(pattern))
                entrenamiento.append((words, intent['tag']))
        return entrenamiento

    def extraer_caracteristicas(self, documento):
        palabras_documento = set(documento)
        caracteristicas = {}
        for palabra in self.palabra_caracteristicas.keys():
            caracteristicas['contiene({})'.format(palabra)] = (palabra in palabras_documento)
        return caracteristicas

    def train_classifier(self):
        entrenamiento = self.preparar_datos(self.intents)
        self.palabra_caracteristicas = nltk.FreqDist([word for words, tag in entrenamiento for word in words])
        entrenamiento_con_caracteristicas = [(self.extraer_caracteristicas(texto), tag) for (texto, tag) in entrenamiento]
        classifier = NaiveBayesClassifier.train(entrenamiento_con_caracteristicas)
        with open('classifier.pkl', 'wb') as file:
            pickle.dump(classifier, file)
        return classifier

    def obtener_respuesta(self, texto_usuario):
        texto_usuario = self.normalizar_texto(texto_usuario)
        palabras = word_tokenize(texto_usuario)
        palabras_filtradas = [w for w in palabras if w.isalnum()]
        caracteristicas = self.extraer_caracteristicas(palabras_filtradas)
        etiqueta = self.classifier.classify(caracteristicas)
        for intent in self.intents['intents']:
            if intent['tag'] == etiqueta:
                return random.choice(intent['responses'])
        return "No entiendo tu pregunta."
