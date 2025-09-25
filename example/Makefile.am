AUTOMAKE_OPTIONS=foreign
AM_LDFLAGS = @OPENMP_CXXFLAGS@
AM_FCFLAGS = -I../SZp/include -I../SZp
AM_CXXFLAGS = -I../SZp/include -I../SZp
if OPENMP
AM_CXXFLAGS+=-fopenmp
endif

bin_PROGRAMS=testfloat_compress_fastmode1 testfloat_decompress_fastmode1 testfloat_multidim_comp_decom testfloat_auto_vectorization

testfloat_compress_fastmode1_SOURCES=testfloat_compress_fastmode1.cc
testfloat_compress_fastmode1_LDADD=../SZp/.libs/libszp.a -lm
testfloat_decompress_fastmode1_SOURCES=testfloat_decompress_fastmode1.cc
testfloat_decompress_fastmode1_LDADD=../SZp/.libs/libszp.a -lm
testfloat_multidim_comp_decom_SOURCES=test_multidim.cc
testfloat_multidim_comp_decom_LDADD=../SZp/.libs/libszp.a -lm
testfloat_auto_vectorization_SOURCES=testfloat_auto_vectorization.cc
testfloat_auto_vectorization_LDADD=../SZp/.libs/libszp.a -lm