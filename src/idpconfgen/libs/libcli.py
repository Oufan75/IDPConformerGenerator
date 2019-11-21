"""Common operations for client interfaces."""
import argparse
import sys


detailed = "detailed instructions:\n\n{}"


class ArgsToTuple(argparse.Action):
    """Convert list of arguments in tuple."""
    
    def __call__(self, parser, namespace, values, option_string=None):
        """Call it."""
        namespace.record_name = tuple(values)


# https://stackoverflow.com/questions/4042452
class CustomParser(argparse.ArgumentParser):
    """Custom Parser class."""
    
    def error(self, message):
        """Present error message."""
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def parse_doc_params(docstring):
    """
    Parse client docstrings.

    Separates PROG, DESCRIPTION and USAGE from client main docstring.

    Parameters
    ----------
    docstring : str
        The module docstring.

    Returns
    -------
    tuple
        (prog, description, usage)
    """
    doclines = docstring.lstrip().split('\n')
    prog = doclines[0]
    description = '\n'.join(doclines[2:doclines.index('USAGE:')])
    usage = '\n'.join(doclines[doclines.index('USAGE:') + 1:])
     
    return prog, description, usage
