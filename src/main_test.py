class Student(object):
    name = ''

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


def main():
    print('这里是main...')


if __name__ == '__main__':
    main()
    stu = Student('菜菜')
    print(stu,stu.get_name())
