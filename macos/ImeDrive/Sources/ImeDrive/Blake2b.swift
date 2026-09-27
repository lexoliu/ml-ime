import Foundation

/// BLAKE2b over a byte stream, first 8 bytes of the digest read big-endian —
/// identical to `EvalRecord::digest` in Rust and `gui.digest` in Python,
/// which the dev/test split keys on.
enum Blake2b {
    private static let iv: [UInt64] = [
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
    ]

    private static let sigma: [[UInt8]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ]

    /// The record's split key: `pinyin + NUL + text + NUL + (context or "")`,
    /// hashed with an 8-byte output and read as a big-endian u64.
    static func digest(pinyin: String, text: String, context: String?) -> UInt64 {
        var data = Data()
        data.append(contentsOf: pinyin.utf8)
        data.append(0)
        data.append(contentsOf: text.utf8)
        data.append(0)
        data.append(contentsOf: (context ?? "").utf8)

        var h = iv
        // Parameter block: digest length 8, key length 0, fanout 1, depth 1.
        h[0] ^= 0x01010000 | 8

        // BLAKE2b-8 hashes at most two 128-byte blocks here in practice, but
        // loop anyway — eval records can exceed 128 bytes of UTF-8.
        let bytes = [UInt8](data)
        var offset = 0
        while bytes.count - offset > 128 {
            compress(&h, block: Array(bytes[offset..<(offset + 128)]),
                     t: UInt64(offset + 128), last: false)
            offset += 128
        }
        var last = [UInt8](repeating: 0, count: 128)
        let tail = bytes.count - offset
        if tail > 0 { last.replaceSubrange(0..<tail, with: bytes[offset...]) }
        compress(&h, block: last, t: UInt64(bytes.count), last: true)

        // The digest's first 8 bytes are h[0] little-endian; Python reads them
        // big-endian, which is just a byte swap.
        return h[0].byteSwapped
    }

    private static func compress(
        _ h: inout [UInt64], block: [UInt8], t: UInt64, last: Bool
    ) {
        var m = [UInt64](repeating: 0, count: 16)
        for i in 0..<16 {
            m[i] = UInt64(littleEndian: block.withUnsafeBytes {
                $0.load(fromByteOffset: i * 8, as: UInt64.self)
            })
        }
        var v = h + iv
        v[12] ^= t
        if last { v[14] = ~v[14] }

        func g(_ a: Int, _ b: Int, _ c: Int, _ d: Int, _ x: UInt64, _ y: UInt64) {
            v[a] = v[a] &+ v[b] &+ x
            v[d] = (v[d] ^ v[a]).rotate(right: 32)
            v[c] = v[c] &+ v[d]
            v[b] = (v[b] ^ v[c]).rotate(right: 24)
            v[a] = v[a] &+ v[b] &+ y
            v[d] = (v[d] ^ v[a]).rotate(right: 16)
            v[c] = v[c] &+ v[d]
            v[b] = (v[b] ^ v[c]).rotate(right: 63)
        }

        for round in 0..<12 {
            let s = sigma[round]
            g(0, 4, 8, 12, m[Int(s[0])], m[Int(s[1])])
            g(1, 5, 9, 13, m[Int(s[2])], m[Int(s[3])])
            g(2, 6, 10, 14, m[Int(s[4])], m[Int(s[5])])
            g(3, 7, 11, 15, m[Int(s[6])], m[Int(s[7])])
            g(0, 5, 10, 15, m[Int(s[8])], m[Int(s[9])])
            g(1, 6, 11, 12, m[Int(s[10])], m[Int(s[11])])
            g(2, 7, 8, 13, m[Int(s[12])], m[Int(s[13])])
            g(3, 4, 9, 14, m[Int(s[14])], m[Int(s[15])])
        }
        for i in 0..<8 { h[i] ^= v[i] ^ v[i + 8] }
    }
}

private extension UInt64 {
    func rotate(right bits: UInt64) -> UInt64 {
        (self >> bits) | (self << (64 - bits))
    }
}
