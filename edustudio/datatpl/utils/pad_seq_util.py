# reference: https://github.com/keras-team/keras/blob/v2.12.0/keras/utils/data_utils.py#L1009-L1143

import numpy as np
from collections import defaultdict
import logging


logger = logging.getLogger("edustudio")


class PadSeqUtil(object):
    @staticmethod
    def pad_sequence(
            sequences, maxlen=0, dtype='int64', padding='post',
            is_truncate=False, truncating='post', value=0.0, 
            return_idx=False, return_mask=False,
        ):
        if not hasattr(sequences, "__len__"):
            raise ValueError("`sequences` must be iterable.")
        num_samples = len(sequences)

        lengths = []
        sample_shape = ()
        flag = True

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.

        for x in sequences:
            try:
                lengths.append(len(x))
                if flag and len(x):
                    sample_shape = np.asarray(x).shape[1:]
                    flag = False
            except TypeError as e:
                raise ValueError(
                    "`sequences` must be a list of iterables. "
                    f"Found non-iterable: {str(x)}"
                ) from e

        if maxlen <= 0 or maxlen is None:
            maxlen = np.max(lengths)


        return_idx_list = []
        if not is_truncate:
            num_samples = 0
            for idx, len_ in enumerate(lengths):
                num = int(np.ceil(len_ / maxlen))
                num_samples += num
                if return_idx:
                    return_idx_list.append([idx] * num)

        if return_idx:
            return_idx = np.concatenate(return_idx_list).astype(np.int64)

        is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
            dtype, np.unicode_
        )
        if isinstance(value, str) and dtype != object and not is_dtype_str:
            raise ValueError(
                f"`dtype` {dtype} is not compatible with `value`'s type: "
                f"{type(value)}\nYou should set `dtype=object` for variable length "
                "strings."
            )

        x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
        mask = None
        if return_mask:
            mask = np.full((num_samples, maxlen) + sample_shape, 1, dtype=np.int8)
        idx = 0
        flag = True
        for s in sequences:
            if not len(s):
                if flag:
                    logger.warning("empty list/array was found")
                    flag = False
                continue

            if is_truncate:
                if truncating == "pre":
                    trunc = s[-maxlen:]
                elif truncating == "post":
                    trunc = s[:maxlen]
                else:
                    raise ValueError(f'Truncating type "{truncating}" not understood')

                # check `trunc` has expected shape
                trunc = np.asarray(trunc, dtype=dtype)
                if trunc.shape[1:] != sample_shape:
                    raise ValueError(
                        f"Shape of sample {trunc.shape[1:]} of sequence at "
                        f"position {idx} is different from expected shape "
                        f"{sample_shape}"
                    )

                if padding == "post":
                    x[idx, : len(trunc)] = trunc
                    if return_mask:
                        mask[idx, len(trunc):] = 0
                elif padding == "pre":
                    x[idx, -len(trunc) :] = trunc
                    if return_mask:
                        mask[idx, :-len(trunc)] = 0
                else:
                    raise ValueError(f'Padding type "{padding}" not understood')
                idx += 1
            else:
                for i in range(int(np.ceil(len(s) / maxlen))):
                    trunc = s[i*maxlen:maxlen*(i+1)]
                    # check `trunc` has expected shape
                    trunc = np.asarray(trunc, dtype=dtype)
                    if trunc.shape[1:] != sample_shape:
                        raise ValueError(
                            f"Shape of sample {trunc.shape[1:]} of sequence at "
                            f"position {idx} is different from expected shape "
                            f"{sample_shape}"
                        )
                    if padding == "post":
                        x[idx, : len(trunc)] = trunc
                        if return_mask:
                            mask[idx, len(trunc):] = 0
                    elif padding == "pre":
                        x[idx, -len(trunc) :] = trunc
                        if return_mask:
                            mask[idx, :-len(trunc)] = 0
                    else:
                        raise ValueError(f'Padding type "{padding}" not understood')
                    idx += 1
        return x, return_idx, mask


if __name__ == '__main__':
    a = PadSeqUtil.pad_sequence(
        [[1],[2,1,3,4,4,4,5,0,5,19,6],[],[0,1,1,1,1,1]], maxlen=4, is_truncate=False, value=-1, return_idx=True, return_mask=True, padding='pre')
    print(a)
