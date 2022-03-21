import time


def getTime():
    return str(time.strftime('%X %x %Z'))


def e(message):
    print(getTime() + "▮▮ e: " + str(message))


def i(message):
    print(getTime() + "▮▮ i: " + str(message))


def d(message):
    print(getTime() + "▮▮ d: " + str(message))


def v(message):
    print(getTime() + "▮▮ v: " + str(message))
