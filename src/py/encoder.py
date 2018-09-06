from enum import Enum
import util
import numpy as np
import pandas as pd
import math
import data
import pickle
from abc import ABC, abstractmethod


class EncodingFactory:
    class _EncodingType(Enum):
        RAW = 'raw'
        FACTOR = 'factor'
        IDF = 'idf'
        PCP = 'pcp'

    @classmethod
    def create(cls, encoding_str, *args):
        encoding_type = EncodingFactory._EncodingType(encoding_str)
        if encoding_type is EncodingFactory._EncodingType.RAW:
            return RAW(*args)
        if encoding_type is EncodingFactory._EncodingType.FACTOR:
            return Factor(*args)
        elif encoding_type is EncodingFactory._EncodingType.IDF:
            return IDF(*args)
        elif encoding_type is EncodingFactory._EncodingType.PCP:
            return PCP(*args)
        else:
            raise AttributeError('Invalid encoding type: {}'.format(encoding_type))


class Encoding(ABC):
    def __init__(self):
        self.values = {}  # Map<Column, Map<Raw_Value, Encoded_Value>>

    @abstractmethod
    def encode(self, column):
        raise NotImplementedError

    @abstractmethod
    def missing_value(self, column_name, value):
        raise NotImplementedError

    def init_column(self, column_name):
        if column_name not in self.values:
            self.values[column_name] = {}

    def exists(self, column, value):
        return column in self.values and value in self.values[column]

    def set(self, column, key, value):
        self.values[column][key] = value

    def get(self, column, key):
        return self.values[column][key]


class Factor(Encoding):
    def __init__(self):
        super().__init__()
        self.ids = {}

    def encode(self, column):
        self.init_column(column.name)
        factor_col = np.zeros(len(column), dtype=np.uint64)
        for i, raw_value in enumerate(column):
            if self.exists(column.name, raw_value):
                encoded_value = self.get(column.name, raw_value)
            else:
                encoded_value = self.set(column.name, raw_value, None)
            factor_col[i] = encoded_value
        return pd.DataFrame({column.name: factor_col})

    def set(self, column, key, value=None):
        assert value is None  # Assigned automatically, sequentially

        id = self.ids[column]
        super(Factor, self).set(column, key, id)
        self.ids[column] += 1
        return id

    def init_column(self, column_name):
        super(Factor, self).init_column(column_name)
        if column_name not in self.ids:
            self.ids[column_name] = 1

    def missing_value(self, column_name, value):
        return 0


class IDF(Encoding):
    def __init__(self, keep_first=False):
        super().__init__()
        self.keep_first = util.str_to_bool(keep_first)
        self.length = 0

    def encode(self, column):
        self.keep_first and self.init_column(column.name)
        tf = util.table_dict(column)
        self.length = len(column)
        idf = {}
        for key, freq in tf.items():
            if self.keep_first and self.exists(column.name, key):
                idf_value = self.get(column.name, key)
            else:
                idf_value = math.log(self.length / freq)
                if self.keep_first:
                    self.set(column.name, key, idf_value)
            idf[key] = idf_value
        idf_col = np.zeros(self.length, dtype=data.INPUTS_DTYPE)
        for i, key in enumerate(column):
            idf_col[i] = idf[key]
        return pd.DataFrame({column.name: idf_col})

    def missing_value(self, column_name, value):
        return math.log(self.length)


class RAW(Encoding):
    def __init__(self):
        super().__init__()

    def encode(self, column):
        encoded_col = column.astype(data.INPUTS_DTYPE)
        return pd.DataFrame({column.name: encoded_col})

    def missing_value(self, column_name, value):
        return float(value)


class PCP(Encoding):
    OTHERS_STR = 'Other'
    SEP = '__'

    def __init__(self, percentage=0.05):
        super().__init__()
        self.percentage = float(percentage)

    def prune(self, column):
        N = len(column)
        frequencies = column.value_counts(ascending=True)
        prune_limit = N * self.percentage
        sum = 0
        to_prune = list()
        for value, freq in frequencies.items():
            sum += freq
            if sum > prune_limit:
                break
            to_prune.append(value)
        column_pruned = column.replace(to_prune, PCP.OTHERS_STR) if to_prune else column
        return column_pruned

    @staticmethod
    def get_others_col_name(column_name):
        return column_name + PCP.SEP + PCP.OTHERS_STR

    @staticmethod
    def feature_split(column_name):
        split = column_name.split(PCP.SEP)
        feature = split[0]
        value = split[1] if len(split) > 1 else None
        if len(split) > 2:
            print('Warning: column name \'{}\' is ambiguous and cannot be accurately split into feature and value. '
                  'Assuming feature = \'{}\', value = \'{}\''.format(column_name, feature, value))
        return feature, value

    def encode(self, column):
        pruned = self.prune(column)
        one_hot = pd.get_dummies(pruned, prefix=pruned.name, prefix_sep=PCP.SEP)
        others_col_name = self.get_others_col_name(column.name)
        if others_col_name not in one_hot:
            one_hot[others_col_name] = 0
        return one_hot

    def missing_value(self, column_name, value):
        return 1


class ValueMapping:
    def __init__(self, encoded_value, col_index):
        self.value = encoded_value
        self.column = col_index


class ColumnMapping:
    def __init__(self, raw_column, encoded_df, encoding):
        assert isinstance(encoding, Encoding)
        self.encoding = encoding
        self.column_name = raw_column.name
        self.default_column = None
        self.values = {}  # Map: raw_value, ValueMapping
        if isinstance(self.encoding, PCP):
            self._pcp_mapping(raw_column, encoded_df)
        else:
            self._direct_mapping(raw_column, encoded_df)

    def _pcp_mapping(self, raw_col, encoded_df):
        for i, col_name in enumerate(encoded_df):
            feature, value = PCP.feature_split(col_name)
            if feature == raw_col.name:
                if value == PCP.OTHERS_STR:
                    self.default_column = i
                else:
                    self.values[value] = ValueMapping(1, i)

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
            return ValueMapping(self.encoding.missing_value(self.column_name, value), self.default_column)

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

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def map(self, input_list):
        if len(input_list) != self.n_cols_raw:
            raise AttributeError('Input length mismatch: expected {}, got {}'.format(self.n_cols_raw, len(input_list)))
        mapped_inputs = np.zeros(self.n_cols_encoded, dtype=data.INPUTS_DTYPE)
        for i, column_name in enumerate(self.col_names_raw):
            value_map = self[column_name][input_list[i]]
            mapped_inputs[value_map.column] = value_map.value
        return mapped_inputs


class Encoder:
    def __init__(self, cfg_file):
        self.columns = {}
        self.default = None

        self._setup(cfg_file)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

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
