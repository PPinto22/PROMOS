from enum import Enum
import util
import numpy as np
import pandas as pd
import math
import data

class EncodingFactory:
    class _EncodingType(Enum):
        RAW = 'raw'
        IDF = 'idf'
        PCP = 'pcp'

    @classmethod
    def create(cls, encoding_str, *args):
        encoding_type = EncodingFactory._EncodingType(encoding_str)
        if encoding_type is EncodingFactory._EncodingType.RAW:
            return RAW()
        elif encoding_type is EncodingFactory._EncodingType.IDF:
            return IDF()
        elif encoding_type is EncodingFactory._EncodingType.PCP:
            return PCP(*args)
        else:
            raise AttributeError('Invalid encoding type: {}'.format(encoding_type))


class Encoding:
    @staticmethod
    def encode(column):
        raise NotImplementedError  # abstract method

    @staticmethod
    def missing_value(length, value):
        raise NotImplementedError  # abstract method


class IDF(Encoding):
    @staticmethod
    def encode(column):
        tf = util.table_dict(column)
        N = len(column)
        idf = {}
        for key, freq in tf.items():
            idf[key] = math.log(N / freq)
        idf_col = np.zeros(N, dtype=data.INPUTS_DTYPE)
        for i, key in enumerate(column):
            idf_col[i] = idf[key]
        return pd.DataFrame({column.name: idf_col})

    @staticmethod
    def missing_value(length, value):
        return math.log(length)


class RAW(Encoding):
    @staticmethod
    def encode(column):
        for i in range(len(column)):
            column[i] = data.INPUTS_DTYPE(column[i])
        return pd.DataFrame({column.name: column})

    @staticmethod
    def missing_value(length, value):
        return float(value)


class PCP(Encoding):
    def __init__(self, percentage=0.05):
        self.percentage = percentage

    @staticmethod
    def encode(column):
        raise NotImplementedError  # TODO

    @staticmethod
    def missing_value(length, value):
        return 1


class ValueMapping:
    def __init__(self, encoded_value, col_index):
        self.value = encoded_value
        self.column = col_index


class ColumnMapping:
    def __init__(self, raw_column, encoded_df, encoding):
        assert isinstance(encoding, Encoding)
        self.encoding = encoding
        self.col_length = len(raw_column)
        self.default_column = None
        self.values = {}  # Map: raw_value, ValueMapping
        if isinstance(self.encoding, PCP):
            self._pcp_mapping(raw_column, encoded_df)
        else:
            self._direct_mapping(raw_column, encoded_df)

    def _pcp_mapping(self, raw_col, encoded_df):
        # TODO
        # self.default_column = 'others' column index
        raise NotImplementedError

    def _direct_mapping(self, raw_col, encoded_df):
        encoded_col_i = list(encoded_df).index(raw_col.name)
        self.default_column = encoded_col_i
        encoded_col = encoded_df[raw_col.name]
        for i, (raw_val, encoded_val) in enumerate(zip(raw_col, encoded_col)):
            if raw_val in self.values:
                if self.values[raw_val].value != encoded_val:
                    raise AttributeError('Raw value \'{}\' has multiple encodings: \'{}\' and \'{}\''.
                                         format(raw_val, self.values[raw_val].value, encoded_val))
            else:
                self.values[raw_val] = ValueMapping(encoded_val, encoded_col_i)

    def get_value(self, value):
        if value in self.values:
            return self.values[value]
        else:
            return ValueMapping(self.encoding.missing_value(self.col_length, value), self.default_column)

    def __getitem__(self, item):
        return self.get_value(item)


class Mapping:
    def __init__(self, raw_df, encoded_df, encoder):
        self.n_cols_raw = len(raw_df.columns)
        self.n_cols_encoded = len(encoded_df.columns)
        self.col_names_raw = list(raw_df.columns)
        self.col_names_encoded = list(encoded_df.columns)
        self.columns = {}  # Map: col_name, ColumnMapping
        for col_name in raw_df:
            self.columns[col_name] = ColumnMapping(raw_df[col_name], encoded_df, encoder[col_name])

    def __getitem__(self, item):
        return self.columns[item]


class Encoder:
    def __init__(self, cfg_file):
        self.columns = {}
        self.default = None

        self._setup(cfg_file)

    def _setup(self, cfg_file):
        with open(cfg_file, 'r') as file:
            lines = [line.strip() for line in file]
            lines = [line for line in lines if line and not line.startswith('#')]

        for line in lines:
            fields = line.split()
            column, encoding, *args = fields
            self.add_column(column, encoding, *args)

    def add_column(self, column, encoding, *args):
        if column in self.columns:
            raise ValueError('Multiple definitions of column \'{}\''.format(column))

        if column == '*':
            if self.default is not None:
                raise ValueError('Multiple definitions of the default encoding')
            self.default = EncodingFactory.create(encoding, *args)
        else:
            self.columns[column] = EncodingFactory.create(encoding, *args)

    def __getitem__(self, item):
        return self.get_encoding(item)

    def get_encoding(self, column):
        if column in self.columns:
            return self.columns[column]
        elif self.default is not None:
            return self.default
        else:
            raise ValueError('Column \'{}\' is not specified and the encoder does not have a default encoding'
                             .format(column))

    def encode_column(self, column):
        encoded_col = self.get_encoding(column.name).encode(column)
        return encoded_col

    def encode(self, df, return_mapping=False):
        encoded_cols = [self.encode_column(df[c]) for c in df]
        encoded_df = pd.concat(encoded_cols, axis=1, join_axes=[encoded_cols[0].index])
        if return_mapping:
            return encoded_df, self.get_mapping(df, encoded_df)
        else:
            return encoded_df

    @staticmethod
    def encode_from_mapping(df, mapping):
        df_encoded = pd.DataFrame(0.0, index=range(len(df)), columns=mapping.col_names_encoded, dtype=data.INPUTS_DTYPE)
        for i in range(len(df)):
            for j, col_name in enumerate(df.columns):
                raw_value = df.iat[i, j]
                value_map = mapping[col_name][raw_value]
                df_encoded.iat[i, value_map.column] = value_map.value
        return df_encoded

    def get_mapping(self, raw_df, encoded_df):
        return Mapping(raw_df, encoded_df, self)
