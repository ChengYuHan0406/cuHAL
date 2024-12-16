add_includedirs("./include/")
add_requires("openmp")
add_requires("gtest")
add_cxxflags("-O3")

target("DesignMatrix")
  set_kind("shared")
  add_files("./src/DesignMatrix.cpp")
  add_packages("openmp")

target("test_DesignMatrix")
  set_kind("binary")
  add_files("./tests/test_DesignMatrix.cpp")
  add_packages("gtest")
  add_deps("DesignMatrix")
  
