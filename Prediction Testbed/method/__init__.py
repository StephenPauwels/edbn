from method.method import Method

from method import sdl
from Predictions import edbn_adapter as edbn
from RelatedMethods.Camargo import adapter as camargo
from RelatedMethods.DiMauro import adapter as dimauro
from RelatedMethods.Lin import adapter as lin
from RelatedMethods.Pasquadibisceglie import adapter as pasquadibisceglie
from RelatedMethods.Tax import adapter as tax
from RelatedMethods.Taymouri import adapter as taymouri

SDL = Method("SDL", sdl.train, sdl.test, {"epochs": 200, "early_stop": 10})
DBN = Method("DBN", edbn.train, edbn.test)
CAMARGO = Method("Camargo", camargo.train, camargo.test, {"epochs": 200, "early_stop": 10})
DIMAURO = Method("Di Mauro", dimauro.train, dimauro.test, {"epochs": 200, "early_stop": 10})
LIN = Method("Lin", lin.train, lin.test, {"epochs": 200, "early_stop": 10})
PASQUADIBISCEGLIE = Method("Pasquadibisceglie", pasquadibisceglie.train, pasquadibisceglie.test, {"epochs": 200, "early_stop": 10})
TAX = Method("Tax", tax.train, tax.test, {"epochs": 200, "early_stop": 10})
TAYMOURI = Method("Taymouri", taymouri.train, taymouri.test, {"epoch": 10})

ALL = [TAX, LIN, CAMARGO, DIMAURO, PASQUADIBISCEGLIE, TAYMOURI, DBN, SDL]