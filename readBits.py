import numpy as np
import matplotlib.pyplot as plt

f1 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_tx"), dtype=np.uint8)
f2 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_tx_crc"), dtype=np.uint8)
f3 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_tx_packed"), dtype=np.uint8)
f4 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_tx_symbols"), dtype=np.uint8)
f5 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_rx_symbols"), dtype=np.uint8)
f6 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_rx_stream"), dtype=np.uint8)
f7 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_rx_crc"), dtype=np.uint8)
f8 = np.fromfile(open("/home/ahmadshokair/test_ofdm/data_rx"), dtype=np.uint8)

b1 = np.unpackbits(f1, axis=0)
b2 = np.unpackbits(f2, axis=0)
b3 = np.unpackbits(f3, axis=0)
b4 = np.unpackbits(f4, axis=0)
b5 = np.unpackbits(f5, axis=0)
b6 = np.unpackbits(f6, axis=0)
b7 = np.unpackbits(f7, axis=0)
b8 = np.unpackbits(f8, axis=0)

print(f'generated random bits:              {b1.size}')
print(f'bits with crc added:                {b2.size}')
print(f'packed bits (represent symbols):    {b3.size}')
print(f'bits from complex IQ symbols:       {b4.size}')
print(f'bits from received complex IQ symbols:{b5.size}')
print(f'packed received bits:               {b6.size}')
print(f'unpacked received bits:             {b7.size}')
print(f'bits after dropping packets:        {b8.size}')

print(f'number of error bits in the received symbols:   {np.count_nonzero(b4[0:b5.size] != b5)}')
print(f'number of error bits in the packed bits:        {np.count_nonzero(b3[0:b6.size] != b6)}')
print(f'number of error bits in the unpacked bits:      {np.count_nonzero(b2[0:b7.size] != b7)}')
print(f'number of error bits in the final data:         {np.count_nonzero(b1[0:b8.size] != b8)}')

errorCount = np.count_nonzero(b2[0:b7.size] != b7)
BER = errorCount/b7.size
print(f'BER is {BER}')
print(f'correctly received packets are {b8.size//8//96} from {b1.size//8//96}, percentage of {b8.size/b1.size*100}%')

plt.figure(1)
plt.plot(b2)
plt.plot(b7)
plt.show()
