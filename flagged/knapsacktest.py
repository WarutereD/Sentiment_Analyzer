import random
import sympy

def generate_private_key():
    # Generate a super-increasing sequence of length 8
    attempts = 0
    while attempts < 1000:
        private_key = [random.randint(1, 100)]
        for i in range(1, 8):
            while True:
                new_element = private_key[i-1] + random.randint(1, 100)
                if new_element > sum(private_key[:i]):
                    private_key.append(new_element)
                    break
        if len(private_key) == 8:
            return private_key
        attempts += 1
    raise RuntimeError("Unable to generate private key after 1000 attempts")




def generate_public_key(private_key, w, m, length=None):
    if length is None:
        length = len(private_key)
    # Generate the public key as a sequence of modular products
    public_key = [(w * private_key[i]) % m for i in range(length)]
    return public_key




def find_prime_above(n):
    # Find the first prime number greater than n
    return sympy.nextprime(n)

def generate_w(m):
    # Generate a random integer less than m that is coprime to m
    while True:
        w = random.randint(2, m-1)
        if sympy.isprime(w) and sympy.mod_inverse(w, m) is not None:
            return w

def encrypt_message(plaintext, public_key):
    if len(plaintext) != len(public_key):
        raise ValueError("Plaintext and public key must have the same length")
        
    # Convert the message to binary and encrypt each 8-bit chunk
    ciphertext = []
    for i in range(0, len(plaintext), 8):
        chunk = plaintext[i:i+8]
        binary_chunk = int(chunk, 2)
        encrypted_chunk = sum([binary_chunk & (1 << j) and public_key[j] for j in range(8)])
        ciphertext.append(encrypted_chunk)
    return ciphertext

def decrypt_message(ciphertext, private_key, w, m):
    if len(ciphertext) != len(private_key):
        raise ValueError("Ciphertext and private key must have the same length")
    # Decrypt each chunk of the ciphertext using the private key
    message = ''
    for i in range(len(ciphertext)):
        decrypted_chunk = (ciphertext[i] * sympy.mod_inverse(w * private_key[i], m)) % m
        binary_chunk = ''.join(['1' if (decrypted_chunk >> j) & 1 else '0' for j in range(8)])
        message += binary_chunk
    return message

# Generate private key, M, and W
private_key = generate_private_key()
M = find_prime_above(sum(private_key))
W = generate_w(M)

# Generate public key
public_key = generate_public_key(private_key, W, M)  # use default length


# Encrypt message
message = 'helloeveryone'
binary_message = ''.join(['{:08b}'.format(ord(c)) for c in message])
public_key = generate_public_key(private_key, W, M, len(binary_message))  # update public_key with correct length
ciphertext = encrypt_message(binary_message, public_key)

# Decrypt message
decrypted_message = decrypt_message(ciphertext, private_key, W, M)
plaintext = ''.join([chr(int(decrypted_message[i:i+8], 2)) for i in range(0, len(decrypted_message), 8)])



print("Original message: ", message)
print("Encrypted message: ", ciphertext)
print("Decrypted message: ", plaintext)
