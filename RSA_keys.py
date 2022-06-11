import random
import sympy.ntheory as nt


class Key:
    def __init__(self, size_of_key):
        self.size_of_key = size_of_key
        self.n = 0
        self.e = 536357  # arbitrarily chosen prime num
        self.d = 0

    # returns pair of p and q
    def generate_pq(self, n):

        p = q = 4
        while not nt.isprime(p):
            p = random.randrange(2**(n-1)+1, 2**n-1)

        while not nt.isprime(q):
            q = random.randrange(2**(n-1)+1, 2**n-1)

        return p, q

    # returns pair of public and private keys
    def get_keys(self, p, q):

        n = p*q
        self.n = n

        fi = (p-1)*(q-1)
        d = pow(self.e, -1, fi)
        self.d = d

        public_key = (self.e, n)
        private_key = (d, n)

        return public_key, private_key
