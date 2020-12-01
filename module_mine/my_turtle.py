import turtle


class Turtle_new(turtle.Turtle):
    def box(self, pos_x, pos_y, x, y, color):
        self.penup()
        self.goto(pos_x - x / 2, pos_y - y / 2)
        self.pendown()
        self.speed('fast')
        self.color(color)
        self.begin_fill()
        self.fd(x)
        self.left(90)
        self.fd(y)
        self.left(90)
        self.fd(x)
        self.left(90)
        self.fd(y)
        self.left(90)
        self.end_fill()

    def print_finish(self):
        print('finish drawing')


class Person:
    def __init__(self, name, age, address):
        self.hello = '안녕하세요'
        self.name = name
        self.age = age
        self.address = address

    def greeting(self):
        print('{0} 저는 {1}입니다'.format(self.hello, self.name))


class Flight:

    def __init__(self):
        print('init')
        super().__init__()

    def __new__(cls):
        print('new')
        return super().__new__(cls)

    def number_make(self):
        return 'SN060'
