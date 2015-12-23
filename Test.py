import cPickle


class A(object):

    def __init__(self):
        self.a = 1

    def methoda(self):
        print(self.a)


class B(object):

    def __init__(self):
        self.b = 2
        a = A()
        self.b_a = a.methoda

    def methodb(self):
        print(self.b)
if __name__ == '__main__':
    b = B()
    with open('best_model1.pkl', 'w') as f:
        cPickle.dump(b, f)
