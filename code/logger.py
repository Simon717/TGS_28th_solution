import datetime

class Logger:
    def __init__(self, path, if_print=True):
        self.path = path
        self.if_print = if_print
        self.identify()

    def identify(self):
        with open(self.path, 'a') as f:
            f.write('\n=========== Start log here ===========' + '\n\n')
            f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

    def write(self, str_in):
        with open(self.path, 'a') as f:
            f.write(str(str_in) + '\n')
            if self.if_print:
                print(str_in)

# check
# log = Logger(path='../log/check.txt')
# log.write('check')
# log.write('this')