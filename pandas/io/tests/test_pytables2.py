import nose
import sys
import os
import warnings
import tempfile
from contextlib import contextmanager

import datetime
import numpy as np

import pandas
from pandas import (Series, DataFrame, Panel, MultiIndex, Categorical, bdate_range,
                    date_range, Index, DatetimeIndex, isnull)
from pandas.io.pytables import (HDFStore, get_store, Term, read_hdf,
                                IncompatibilityWarning, PerformanceWarning,
                                AttributeConflictWarning, DuplicateWarning,
                                PossibleDataLossError, ClosedFileError)
from pandas.io import pytables as pytables
import pandas.util.testing as tm
from pandas.util.testing import (assert_panel4d_equal,
                                 assert_panel_equal,
                                 assert_frame_equal,
                                 assert_series_equal)
from pandas import concat, Timestamp
from pandas import compat
from pandas.compat import range, lrange, u
from pandas.util.testing import assert_produces_warning

try:
    import tables
except ImportError:
    raise nose.SkipTest('no pytables')

from distutils.version import LooseVersion

_default_compressor = LooseVersion(tables.__version__) >= '2.2' \
    and 'blosc' or 'zlib'

_multiprocess_can_split_ = False

# contextmanager to ensure the file cleanup
def safe_remove(path):
    if path is not None:
        try:
            os.remove(path)
        except:
            pass


def safe_close(store):
    try:
        if store is not None:
            store.close()
    except:
        pass


def create_tempfile(path):
    """ create an unopened named temporary file """
    return os.path.join(tempfile.gettempdir(),path)


@contextmanager
def ensure_clean_store(path, mode='a', complevel=None, complib=None,
              fletcher32=False):

    try:

        # put in the temporary path if we don't have one already
        if not len(os.path.dirname(path)):
            path = create_tempfile(path)

        store = HDFStore(path, mode=mode, complevel=complevel,
                         complib=complib, fletcher32=False)
        yield store
    finally:
        safe_close(store)
        if mode == 'w' or mode == 'a':
            safe_remove(path)


@contextmanager
def ensure_clean_path(path):
    """
    return essentially a named temporary file that is not opened
    and deleted on existing; if path is a list, then create and
    return list of filenames
    """
    try:
        if isinstance(path, list):
            filenames = [ create_tempfile(p) for p in path ]
            yield filenames
        else:
            filenames = [ create_tempfile(path) ]
            yield filenames[0]
    finally:
        for f in filenames:
            safe_remove(f)


# set these parameters so we don't have file sharing
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS   = 1
tables.parameters.MAX_THREADS   = 1

def _maybe_remove(store, key):
    """For tests using tables, try removing the table to be sure there is
    no content from previous tests using the same table name."""
    try:
        store.remove(key)
    except:
        pass


def compat_assert_produces_warning(w,f):
    """ don't produce a warning under PY3 """
    if compat.PY3:
        f()
    else:
        with tm.assert_produces_warning(expected_warning=w):
            f()


class TestHDFStore(tm.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestHDFStore, cls).setUpClass()

        # Pytables 3.0.0 deprecates lots of things
        tm.reset_testing_mode()

    @classmethod
    def tearDownClass(cls):
        super(TestHDFStore, cls).tearDownClass()

        # Pytables 3.0.0 deprecates lots of things
        tm.set_testing_mode()

    def setUp(self):
        warnings.filterwarnings(action='ignore', category=FutureWarning)

        self.path = 'tmp.__%s__.h5' % tm.rands(10)

    def tearDown(self):
        pass

    def xtest_select_iterator_8014_0(self):

        # single table
        with ensure_clean_store(self.path) as store:

            df = tm.makeTimeDataFrame(10064, 'S')
            _maybe_remove(store, 'df')
            i = 0; ln = 5927
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 4137
            store.append('df', df[i:i+ln])
            expected = df

            beg_dt = expected.index[0]
            end_dt = expected.index[-1]

            # select w/o iteration and no where clause works
            result = store.select('df')
            tm.assert_frame_equal(expected, result)

            # select w/iterator and no where clause works
            results = []
            for s in store.select('df',iterator=True):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, begin
            # of range, works
            where = "index >= '%s'" % beg_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, end
            # of range, fails
            where = "index <= '%s'" % end_dt
            result = store.select('df',where=where)
            #tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, inclusive range,
            # fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            result = store.select('df',where=where)
            #tm.assert_frame_equal(expected, result)

            #
            # with iterator
            #

            # select w/iterator and where clause, single term, begin
            # of range, fails
            where = "index >= '%s'" % beg_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e2):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, single term, end of
            # range, fails
            where = "index <= '%s'" % end_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e2):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, inclusive range, fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e2):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

    def test_select_iterator_8014_1(self):

        # single table
        with ensure_clean_store(self.path) as store:

            df = tm.makeTimeDataFrame(100064, 'S')
            _maybe_remove(store, 'df')
            i = 0; ln = 58689
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 41375
            store.append('df', df[i:i+ln])
            expected = df

            beg_dt = expected.index[0]
            end_dt = expected.index[-1]

            #
            # w/o iterator
            #
            
            # select w/o iteration and no where clause works
            result = store.select('df')
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, begin
            # of range, works
            where = "index >= '%s'" % beg_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, end
            # of range, works
            where = "index <= '%s'" % end_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, inclusive range,
            # works
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            #
            # with iterator
            #

            # select w/iterator and no where clause works
            results = []
            for s in store.select('df',iterator=True):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, single term, begin
            # of range, fails
            where = "index >= '%s'" % beg_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, single term, end of
            # range, fails
            where = "index <= '%s'" % end_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, inclusive range, fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

    def xtest_select_iterator_8014_2(self):

        # single table
        with ensure_clean_store(self.path) as store:

            df = tm.makeTimeDataFrame(200064, 'S')
            _maybe_remove(store, 'df')
            i = 0; ln = 100000
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 58689
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 41375
            store.append('df', df[i:i+ln])
            expected = df

            beg_dt = expected.index[0]
            end_dt = expected.index[-1]

            # select w/o iteration and no where clause works
            result = store.select('df')
            tm.assert_frame_equal(expected, result)

            # select w/iterator and no where clause works
            results = []
            for s in store.select('df',iterator=True):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, begin
            # of range, works
            where = "index >= '%s'" % beg_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, end
            # of range, works
            where = "index <= '%s'" % end_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, inclusive range,
            # fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            #
            # with iterator
            #

            # select w/iterator and where clause, single term, begin
            # of range, works
            where = "index >= '%s'" % beg_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, single term, end of
            # range, fails
            where = "index <= '%s'" % end_dt
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, inclusive range, fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            results = []
            for s in store.select('df',where=where,iterator=True, chunksize=1e4):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

    def xtest_select_iterator_8014_3(self):

        # single table
        with ensure_clean_store(self.path) as store:

            df = tm.makeTimeDataFrame(300064, 'S')
            _maybe_remove(store, 'df')
            i = 0; ln = 200000
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 58689
            store.append('df', df[i:i+ln])
            i = i+ln; ln = 41375
            store.append('df', df[i:i+ln])
            expected = df

            beg_dt = expected.index[0]
            end_dt = expected.index[-1]

            # select w/o iteration and no where clause works
            result = store.select('df')
            tm.assert_frame_equal(expected, result)

            # select w/iterator and no where clause works
            results = []
            for s in store.select('df',iterator=True):
                results.append(s)
            result = concat(results)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, begin
            # of range, works
            where = "index >= '%s'" % beg_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, single term, end
            # of range, works
            where = "index <= '%s'" % end_dt
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            # select w/o iterator and where clause, inclusive range,
            # works
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            result = store.select('df',where=where)
            tm.assert_frame_equal(expected, result)

            #
            # with iterator
            #

            # select w/iterator and where clause, single term, begin
            # of range, fails
            where = "index >= '%s'" % beg_dt
            results = []
            for s in store.select('df',where=where,iterator=True):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, single term, end of
            # range, fails
            where = "index <= '%s'" % end_dt
            results = []
            for s in store.select('df',where=where,iterator=True):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)

            # select w/iterator and where clause, inclusive range, fails
            where = "index >= '%s' & index <= '%s'" % (beg_dt, end_dt)
            results = []
            for s in store.select('df',where=where,iterator=True):
                results.append(s)
            result = concat(results)
            #tm.assert_frame_equal(expected, result)


def _test_sort(obj):
    if isinstance(obj, DataFrame):
        return obj.reindex(sorted(obj.index))
    elif isinstance(obj, Panel):
        return obj.reindex(major=sorted(obj.major_axis))
    else:
        raise ValueError('type not supported here')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
