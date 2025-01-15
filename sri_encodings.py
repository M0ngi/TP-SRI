def gamma_encode(number):
    binary = bin(number)[2:]
    length = len(binary)

    unary = "1" * (length - 1) + "0"

    offset = binary[1:]

    return unary + offset


def gamma_decode(encoded):
    unary_length = encoded.find("0")

    offset = encoded[unary_length + 1 :]

    binary = "1" + offset

    return int(binary, 2)


def gap_encode(doc_ids):
    if not doc_ids:
        return []
    postings = [doc_ids[0]]
    for i in range(1, len(doc_ids)):
        postings.append(doc_ids[i] - doc_ids[i - 1])

    padded_postings = [posting + 1 for posting in postings]

    return padded_postings


def gap_decode(postings):
    if not postings:
        return []

    doc_ids = [postings[0]]
    for i in range(1, len(postings)):
        doc_ids.append(doc_ids[-1] + postings[i])

    return doc_ids


def compress_gamma_binary(gap_encoded):
    if not gap_encoded:
        return b""

    compressed = "".join(gamma_encode(delta) for delta in gap_encoded)

    # Ensure the binary string has a length that is a multiple of 8
    padded_compressed = compressed.ljust((len(compressed) + 7) // 8 * 8, "0")
    binary_compressed = int(padded_compressed, 2).to_bytes(
        (len(padded_compressed) + 7) // 8, byteorder="big"
    )
    return binary_compressed


def decompress_gamma_binary(compressed):
    if not compressed:
        return []

    binary_string = bin(int.from_bytes(compressed, byteorder="big"))[2:].zfill(
        8 * len(compressed)
    )
    decoded = []
    index = 0

    while index < len(binary_string):
        if binary_string[index:].find("1") == -1:
            decoded.append(0)
            break

        unary_length = binary_string[index:].find("0") + 1
        if unary_length == 0:
            break
        index += unary_length

        offset = binary_string[index : index + unary_length - 1]
        index += unary_length - 1

        decoded_value = int("1" + offset, 2) - 1
        decoded.append(decoded_value)

    return decoded