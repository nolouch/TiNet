'''
解码key
'''
import sys
enc_group_size = 8
enc_mark = chr(0xFF)
enc_pad = chr(0x0)
key = "7480000000000000FF675F698000000000FF0000010380000000FF0008F01A03800000FF0000EF6B01000000FC"
key = "7480000000000000FF675F698000000000FF0000010380000000FF0008F01A03800000FF0000EF6B01000000FC"
sign_mask = 0x8000000000000000

def bytes_to_int(key):
    if len(key) != 8:
        raise("insufficient bytes to decode value")
    u = int.from_bytes(key, byteorder="big", signed=False)
    return u ^ sign_mask



def decode_table_prefix(key):
    if len(key) == 9 and chr(key[0])=='t':
        return bytes_to_int(key[1:9])
    return 0

def decode_table_row(key):
    if len(key) == 19 and chr(key[0])=='t' and chr(key[9])=='_' and chr(key[10])=='r':
        table_id = decode_table_prefix(key[0:9])
        row_id = bytes_to_int(key[11:19])
        return (table_id,row_id)
    return (0,0)

def decode_table_index(key):
    if len(key) >= 19 and chr(key[0])=='t' and chr(key[9])=='_' and chr(key[10])=='i':
        table_id = decode_table_prefix(key[0:9])
        index_id = bytes_to_int(key[11:19])
        index_values = key[19:]
        return (table_id, index_id, index_values)
    return (0,0,"")


def decode(key):
    _un_hex_key = bytes.fromhex(key)
    buf = b''
    while 1: 
       if len(_un_hex_key) < enc_group_size+1:
           raise("insufficient bytes to decode")
       group_bytes = _un_hex_key[0:enc_group_size+1]
       group = group_bytes[0:enc_group_size]
       marker = group_bytes[enc_group_size]
       pad_count = ord(enc_mark) - marker
       if pad_count >  enc_group_size:
           raise("invalid marker byte")
       read_group_size = enc_group_size - pad_count
       buf = buf + group[:read_group_size]
       _un_hex_key = _un_hex_key[enc_group_size+1:]
       if pad_count != 0:
           pad_byte = enc_pad
           for v in group[read_group_size:]:
                if v != ord(pad_byte):
                    raise("invalid padding byte")
           break
    key = buf
    table_id = decode_table_prefix(key)
    if table_id:
        return (1, table_id, 2, 1)
    table_id, row_id = decode_table_row(key)
    if table_id:
        return (2, table_id, row_id,2)
    table_id, index_id, index_values = decode_table_index(key)
    if table_id:
        #return (3, table_id, index_id, index_values)
        return (3, table_id, index_id, 3)
    return (table_id)




#print(decode(sys.argv[1]))
    
