####### 더하기 print
class my_class:
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = in_channels
        self.outc = out_channels
        self.do_sum = self.inc + self.outc
        self.temp_sum = sum([self.inc, self.outc])

    def check_sum(self):
        print('start_game')
        print('do_sum : {0} + {1} = {2}'.format(self.inc, self.outc, self.do_sum))
        print('temp_sum : {0} + {1} = {2}'.format(self.inc, self.outc, self.temp_sum))
        print('check : {0}'.format(self.do_sum == self.temp_sum))
        print('end_game')

###### 문자열 print
class Person:
    def __init__(self, name, age, address):
        self.hello = '안녕하세요'
        self.name = name
        self.age = age
        self.address = address

    def greeting(self):
        print('{0} 저는 {1}에 사는 {2}살 {3}입니다'.format(self.hello, self.address, self.age, self.name))


########## matmul 연산해보기
import numpy as np

class my_matrix:

    def __init__(self, matrix1, matrix2, matrix3):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.matrix3 = matrix3

        self.matrix12 = 0
        self.matrix123 = 0

    # shape 확인
    def shape(self):
        print('matrix1.shape : {0}, matrix2.shape : {1}, matrix3.shape : {2}'.format(self.matrix1.shape, self.matrix2.shape, self.matrix3.shape))

    # weight 곱하기, bias 더하기
    def Linear(self):
        self.matrix12 = np.matmul(self.matrix1, self.matrix2)
        print(self.matrix12)
        self.matrix123 = self.matrix12 + self.matrix3
        print(self.matrix123)


############ 부모, 자식 클래스 상속 받기
class father():
    def __init__(self, who):
        self.who = who

    def handsome(self):
        print('{}를 닮아 잘 생겼다.'.format(self.who))

class mother():
    def __init__(self, whom):
        self.whom = whom

    def kind(self):
        print('{}를 닮아 참 착하다.'.format(self.whom))

class sister(father, mother):
    def __init__(self, who, where, whom):

        # 얘를 사용하면 오버로딩 때문에 최종은 whom 으로 받게 된다.
        #super().__init__(who)
        #super().__init__(whom)

        # 상속 받을 다중 class를 직접 입력해주기
        father.__init__(self, who)
        mother.__init__(self, whom)
        self.where = where

    def choice(self):
        print('{} 부분을 닮았네'.format(self.where))

    def handsome(self):
        super().handsome()
        self.choice()

    def kind(self):
        super().kind()

