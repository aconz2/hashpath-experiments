The functions `xform_simd` transforms a 32 byte SHA256 digest into a 40 byte valid pathname (on most Linux filesystems) and `xform_invert_simd` does the reverse. Compared to hex which is 64 bytes and base64 which is 44 bytes.

There is a longer [blog post](https://aconz2.github.io/2026/01/08/Hash-Filenames.html) here.

# Results

On a 5950x, I get:

```
reg  acc=ffffffffe65eb880 elapsed=254464798 per_iter=25.45
simd acc=ffffffffe65eb880 elapsed=21261869 per_iter=2.13
invert  acc=989680 elapsed=22451161 per_iter=2.25
hex  acc=1d34ce80 elapsed=26909584 per_iter=2.69
hexdec  acc=ffffffffccbd7301 elapsed=160562339 per_iter=16.06
binary_to_base64  acc=1a39de00 elapsed=105438762 per_iter=10.54
base64_to_binary  acc=0 elapsed=369681233 per_iter=36.97
```
