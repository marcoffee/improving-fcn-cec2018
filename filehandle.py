import sys


class Null (object):
    """ Null file descriptor (avoids opening os.devnull) """

    def write (*_):
        """ Ignore write """
        pass

    def flush (*_):
        """ Ignore flush """
        pass

    def close (*_):
        """ Ignore close """
        pass

null = Null()

class Log (object):
    """ Log file descriptor (prints to file and sys.stdout) """

    __slots__ = [ "file" ]

    def __init__ (self, file = None):
        """ Constructs log file descriptor """
        self.file = open(file, "w") if file else None

    def write (self, *args):
        """ Write to file and sys.stdout """
        sys.stdout.write(*args)
        self.file and self.file.write(*args)

    def flush (self, *args):
        """ Flush file and sys.stdout """
        sys.stdout.flush(*args)
        self.file and self.file.flush(*args)

    def close (self, *args):
        """ Close file """
        self.file and self.file.close(*args)

log = Log()
