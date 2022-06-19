import math
import zlib  # decompressing if needed
import struct  # parsing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from RSA_keys import Key
from RSA_keys import nt
import sys
'''
    tornado.png -> with PLTE chunk
    dice.png -> with tIME, gAMA
'''

# file_name = 'PNGs/' + 'tornado_encrypted.png'  # new encrypted file

file_name = 'PNGs/' + 'dice.png'

# file_name = 'PNGs/' + 'dice.png'
# file_name = 'Screen_color_test_VGA_256colors.png'
# file_name = 'Screen_color_test_VGA_4colors.png'

file = open(file_name, 'rb')

# image showing
img_show = Image.open(file_name)
# img_show.show()


# Checking if it is a valid PNG file that we loaded
PNG_sig = b'\x89PNG\r\n\x1a\n'
if file.read(len(PNG_sig)) != PNG_sig:
    print("Not valid PNG file")

"""
Chunks' format:
1. (4b) Length of 2. and 3. (can be 0)
2. (4b) Type
3. Data (Length-len(Type)
4. (4b) CRC sum
"""


# Function for chunks reading
def chunk_reading(file):
    # getting length and type
    # >I4s is big endian int32(4) and 4*char(4)
    len_of_chunk, type_of_chunk = struct.unpack('>I4s', file.read(8))
    # getting data
    data_of_chunk = file.read(len_of_chunk)
    # calculating crc based on data
    checksum = zlib.crc32(data_of_chunk, zlib.crc32(struct.pack('>4s', type_of_chunk)))
    crc_of_chunk, = struct.unpack('>I', file.read(4))
    if crc_of_chunk != checksum:
        print(f'Calculated CRC: {checksum} is not equal to the one in the file: {crc_of_chunk}')
    return type_of_chunk, data_of_chunk, len_of_chunk, crc_of_chunk


list_of_chunks = []
PLTE_present = False
tEXt_present = False
tEXt_data = []
index = 0
tIME_present = False
gAMA_present = False


while 1:
    ch_type, data, len, crc = (chunk_reading(file))
    list_of_chunks.append((ch_type, data, len, crc))
    if ch_type == b'PLTE':
        PLTE_present = True
    elif ch_type == b'tEXt':
        tEXt_present = True
        tEXt_data.insert(0, data)
        index += 1
    if ch_type == b'tIME':
        tIME_present = True
    elif ch_type == b'gAMA':
        gAMA_present = True
    elif ch_type == b'IEND':
        index -= 1
        break

list_of_chunk_types = []

size_IDAT = 0
# chunks showing:
for ch_type, data, len, crc in list_of_chunks:
    print(f'Chunk: {ch_type}, of length: {len},data: {data}')
    list_of_chunk_types.append(ch_type)



######################
####### IHDR #########
# IHDR is always first
IHDR_data = list_of_chunks[0][1]

# IHDR is always 13bytes long, contains these objects:
width, height, bit_depth, color_type, compression_m,\
filter_m, interlace_m = struct.unpack('>IIBBBBB', IHDR_data)

# Printing values:
print(f'\nIHDR CHUNK metadata:\nwidth: {width}, height: {height},\nbit_depth: {bit_depth}, color type: {color_type},\n'
      f'compression method: {compression_m}, filter method: {filter_m}, interlace method: {interlace_m}')

##################################
from itertools import zip_longest


def sqrt_int(X: int):
    N = math.floor(math.sqrt(X))
    while bool(X % N):
        N -= 1
    M = X // N
    return M, N


if PLTE_present:
    PLTE_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'PLTE')
    for chunk_type, chunk_data, len, crc in list_of_chunks:
        if chunk_type == b'PLTE' and len % 3 == 0:
            PLTE_len = len

    print(" \ndane do palety barw: ", PLTE_data)
    print("Dlugosc bloku PLTE: "+ str(PLTE_len))
    pixels = struct.unpack('>'+str(PLTE_len)+'B', PLTE_data)
    print(pixels[:12]) # gdy niepogrupowane w trójki

    m, n = sqrt_int(PLTE_len/3)
    pixels = np.reshape(pixels, (int(m), int(n), 3))
    print(pixels.shape)
    fig_1 = plt.figure(1)
    plt.imshow(pixels)
    plt.title('PLTE palette')

########### Optional chunks ##########

##### tEXt chunk ##########
if tEXt_present:
    while index >= 0:
        try:
            key, text = tEXt_data[index].split(b'\x00', 1)
            key = key.decode('utf-8', 'replace')
            text = text.decode('utf-8', 'replace')
        except ValueError:
            key = None
            text = tEXt_data.decode('utf-8', 'replace')
        print(f'\n(*) tEXt info, {key}: {text}')
        index -= 1


##### tIME chunk ##########

if tIME_present:
    tIME_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'tIME')
    year, mon, day, h, min, sec = struct.unpack('>h5B', tIME_data)

    print(f'\n(*) tIME info, last modification: {day}-{mon}-{year}, {h}:{min}:{sec}')

##### gAMA chunk ##########

if gAMA_present:
    gAMA_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'gAMA')
    gamma, = struct.unpack('>I', gAMA_data)

    print(f'\n(*) gAMA info: Gamma = {gamma/100000}')

print("\nChunk types:", list_of_chunk_types)


# Anonymization of PNG
main_chunks = [b'IHDR', b'IDAT', b'IEND']
if PLTE_present:
    main_chunks.insert(1, b'PLTE')
print(main_chunks)

new_file_name = file_name[:-4] + "_anon.png"
new_file_handler = open(new_file_name, 'wb')
new_file_handler.write(PNG_sig)

for chunk in list_of_chunks:
    if chunk[0] in main_chunks:
        new_file_handler.write(struct.pack('>I', chunk[2]))
        new_file_handler.write(chunk[0])
        new_file_handler.write(chunk[1])
        new_file_handler.write(struct.pack('>I', chunk[3]))

new_file_handler.close()


'''
# FFT function
def fft(image):
    img = cv2.imread(image, 0)

    fourier = np.fft.fft2(img)
    # shifting fft to be centred (składowa stała na środku a nie w rogu)
    fourier_shifted = np.fft.fftshift(fourier)
    # składowa stała >>> od pozostałych f modułu, więc zmiana na skalę log
    fourier_mag = np.asarray(20 * np.log10(np.abs(fourier_shifted)), dtype=np.uint8)
    # aby uzysać fazę użyjemy fcji angle z numpy, która zwraca wartości kątów z fft,
    # czyli fazę z liczby urojonej
    fourier_phase = np.asarray(np.angle(fourier_shifted), dtype=np.uint8)

    fig_2 = plt.figure(2)  # show source image and FFT
    plt.subplot(221), plt.imshow(img, cmap="Greys")
    plt.title('Input Image')

    plt.subplot(222), plt.imshow(fourier_mag, cmap="Greys_r")
    plt.title('FFT Magnitude')

    plt.subplot(223), plt.imshow(fourier_phase, cmap="Greys_r")
    plt.title('FFT Phase')

    # checking if FFT is correct
    fig_3 = plt.figure(3)
    fourier_inverted = np.fft.ifft2(fourier)  # inverting fft

    plt.subplot(121), plt.imshow(img, cmap="Greys")
    plt.title('Input Image')
    plt.subplot(122), plt.imshow(np.asarray(fourier_inverted, dtype=np.uint8), cmap="Greys")
    plt.title('Inverted Image')

    plt.show()


fft(file_name)
'''


'''
      RSA part
'''

# rsa_keys = Key(1000)  # size of key for the future (? IDAT size)

# p, q = rsa_keys.generate_pq(2000)  # set to 2000 for good bitlength

# print(f'My keys:\n p of bitcount:{p.bit_length()}: {p},'
      # f'\n q of bitcount:{q.bit_length()}: {q} ')

# print(nt.isprime(q), nt.isprime(p))

# public_keys, private_keys = rsa_keys.get_keys(p, q)


# example keys to remember (don't have to generate all over again)
# They are 4000 bit long
e = 536357
n = 7303485625255718087810243101215839334898553660596918512774136516725077749135609446454876232746626319553232007775995541610513764890167330936049735520247291749458950590854163346152745085728900545020168058392882861954548199812180128759697093808023636490284012229575465368457792287216771498352681721791508361022126142498899524457382129048429270486475813840674579603202847712234677702499372799504389713468248732094587673512820897794048428847332748316244828464014141813099435024588152341705875608350499921703420297299234999746103099805415024687897584772526401010143436651685006163200682172395554371131117009006376923847282756277342434983484024411334996789182274990751337669890133407338591385758952290454698695377548695575781892036651263448491063308131386509375737243080934714698878237410316885740886002262872422175341961507623280138155636979477282849001872975296626455263322983177946629798268978282228596030189707257803004960383178862826563517852895672387296790201929244879196909672111131201619802225928333599183508356156192487847030605579737959923644327702316490300183706928128584644215400027004936185081601607510183142680532453102309237853466332962128283389852144109380913653587167537643005623538838511640107
d = 6713100441032873659317301440830284292187828171673495128799753341049829367984114045500019544340964647687535316652091427936958566944875323994042250984851348695893337164036458048749812768854229493965666249881499171155764281080334186891437358414928215329920217372348462734055287052462946039089397712425145233461869963945576296305426030835573378085552302334923507560037445063887851015894620169319434870319295963178688304695978056802588342133569702492758928163068575433634727368379566416511757420741775461865485500456827924078233020551739992525749657957023989055798123766969962242420507816605373482526825762232564387073150442426295223562866090044662935024254334487283738437402406726666207150577317142471283970090824147961914582003515052960765526727225212192777817910856896501141433509764984874031393446801960069135920390085687432262985011833928025407635511854408813878966536131586236211217937389610117828372711053822875597993635744030710108748194403964485273231913413771248755487980733823800781744598196844265244935609081962333729162288105482069761348939220302716082130952561245071267685551186161686695242334854675121251694552244413040952764005521617584399622051436910865462297953810634026305695749117332797

public_keys = e, n
private_keys = d, n

# Keys showing:
# print(f'\nPublic keys: \ne is {nt.isprime(e)} = {e},\n'
#       f'n is {nt.isprime(n)} = {n},\n'
#       f'Private keys: \nd is {nt.isprime(d)} = {d},\n'
#       f'n is {nt.isprime(n)} = {n},\n')

list_of_len = []
for ch_type, data, len, crc in list_of_chunks:
    print(ch_type, len)
    list_of_len.append(len)

IDAT_crc = list(crc for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'IDAT')
IDAT_len = list(len for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'IDAT')
IDAT_data = b''.join(chunk_data for chunk_type, chunk_data, len, crc in list_of_chunks if chunk_type == b'IDAT')

print(f'\nIDAT of length: {IDAT_len}\n')




import rsa as RSA
# TODO Functions to move to class RSA
def encrypting(data, e, n):
    c = pow(int(data), e, n)  # bytes->int to be able to pow()
    return int(c)


def decrypting(c, d, n):
    m = pow(c, d, n)
    return m


def RSA_comprison(message):
    # message = message.encode('utf8')
    (pub_keys, priv_keys) = RSA.newkeys(512)

    cipher_from_library = RSA.encrypt(message, pub_keys)
    cipher_ours = encrypting(int.from_bytes(message, "big"), pub_keys.e, pub_keys.n)

    decrypted_library = RSA.decrypt(cipher_from_library, priv_keys)
    decrypted_ours = decrypting(cipher_ours, priv_keys.d, priv_keys.n)
    print(f'-------- * ---------\n'
          f'Comparison of two methods:\n'
          f'Message for encryption:\n   {int.from_bytes(message, "big")}\n'
          f'Message after encryption using RSA from library:\n'
          f'    {int.from_bytes(cipher_from_library, "big")}\n'
          f'Message after encryption using our RSA:\n'
          f'    {cipher_ours}\n'
          f'After encryption:\nlibrary: {int.from_bytes(decrypted_library, "big")}, ours: {decrypted_ours}\n'
          f'Are the the same after encryption? {cipher_from_library==cipher_ours} and after decryption? '
          f'{int.from_bytes(decrypted_library, "big")==decrypted_ours}\n'
          f'-------- * ---------\n')

bytes_array = bytearray(IDAT_data)  # bytes array for IDAT data
RSA_comprison(bytes_array[10:20])


# Padding a block of data
def padding(seq, num_bits):
    pad_size = num_bits - sys.getsizeof(seq)
    tmp_zeros = []
    for _ in range(pad_size):
        tmp_zeros.append(b'0')

    return tmp_zeros + seq

print(f'\n!!!!!!!!!!!!size before padding: {sys.getsizeof(bytes_array)}\n')

IDAT_size = sys.getsizeof(IDAT_data)
print(f'Size of all data from IDAT: {IDAT_size}\n')

blockSize = 500  # lesser than size of a key
amountBlocks = int(IDAT_size/blockSize) + 1
cipher_data = bytearray()
tmp_IDAT = bytearray()

for i in range(0, amountBlocks):
    endOfBlock = (i + 1) * blockSize
    if IDAT_size < (i+1)*blockSize: # ten rozmiar trzeba jakoś inaczej ogarnąć, ale może być dobrze
        endOfBlock = (i+1)*blockSize - ((i+1)*blockSize - IDAT_size)
        for _ in range((i+1)*blockSize - IDAT_size):
            tmp_IDAT = b'0' + tmp_IDAT
        print(tmp_IDAT)
        #tmp_IDAT = padding(tmp_IDAT, amountBlocks*blockSize)  # padding

    tmp_IDAT = tmp_IDAT + IDAT_data[i * blockSize:endOfBlock]
    print("tmp_IDAT")
    print(tmp_IDAT)
    print(endOfBlock)
    cipher_int = encrypting(int.from_bytes(tmp_IDAT, 'big'), e, n)
    print("cipher_int")
    print(cipher_int)
    cipher_hex = cipher_int.to_bytes(blockSize+1, 'big')
    print("cipher_hex")
    print(cipher_hex)
    tmp_IDAT = bytearray()
    cipher_data = cipher_data + cipher_hex
    #for j in range(blockSize):
     #   cipher_data.append(cipher_hex[j])

total_sizex = sys.getsizeof(cipher_data)
total_size = blockSize * amountBlocks

example_data_forRSA = bytes_array[:1]
print("Cipher Data:")
print(cipher_data[:100])
print(f'\nlen of ex data: {sys.getsizeof(example_data_forRSA)}\n')


# getting int from bytes
binary_example_dataRSA = int.from_bytes(example_data_forRSA, "big")

c = encrypting(binary_example_dataRSA, e, n)
m = decrypting(c, d, n)

print(f'data for encryption:\n{binary_example_dataRSA}\n'
      f'after encryption:\n{(c)}\n'
      f'after decryption:\n{(m)}')

print(f'\n!!!!!!!! {blockSize}, {amountBlocks} size after padding: {sys.getsizeof(bytes_array)}\n')

#####
# Creating encrypted PNG
#####
main_chunks = [b'IHDR', b'IDAT', b'IEND']
# if PLTE_present:
#     main_chunks.insert(1, b'PLTE')
print(main_chunks)

new_file_name = file_name[:-4] + "_encrypted.png"
new_file_handler = open(new_file_name, 'wb')
new_file_handler.write(PNG_sig)

for chunk in list_of_chunks:
    if chunk[0] in main_chunks:
        if chunk[0] != b'IDAT':
            new_file_handler.write(struct.pack('>I', chunk[2]))
            new_file_handler.write(chunk[0])
            new_file_handler.write(chunk[1])
            new_file_handler.write(struct.pack('>I', chunk[3]))
        else:
            new_file_handler.write(struct.pack('>I', total_size))  # get len
            new_file_handler.write(chunk[0])
            new_file_handler.write(bytes(cipher_data))  # insert data after encryption
            new_file_handler.write(struct.pack('>I', zlib.crc32(bytes(cipher_data), zlib.crc32(struct.pack('>4s', b'IDAT')))))


new_file_handler.close()

# # Creating decrypted PNG
# main_chunks = [b'IHDR', b'IDAT', b'IEND']
# if PLTE_present:
#     main_chunks.insert(1, b'PLTE')
# print(main_chunks)
#
# new_file_name = file_name[:-14] + "_decrypted.png"
# new_file_handler = open(new_file_name, 'wb')
# new_file_handler.write(PNG_sig)
#
# for chunk in list_of_chunks:
#     if chunk[0] != b'IDAT':
#         new_file_handler.write(struct.pack('>I', chunk[2]))
#         new_file_handler.write(chunk[0])
#         new_file_handler.write(chunk[1])
#         new_file_handler.write(struct.pack('>I', chunk[3]))
#     else:
#         new_file_handler.write(struct.pack('>I', chunk[2]))
#         new_file_handler.write(chunk[0])
#         new_file_handler.write("""decrypted_data""")  # insert data after decryption
#         new_file_handler.write(struct.pack('>I', chunk[3]))
#
# new_file_handler.close()
