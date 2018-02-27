#----------------------------------------------------------------
# Generated CMake target import file for configuration "".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libtesseract" for configuration ""
set_property(TARGET libtesseract APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(libtesseract PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "pthread;lept"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtesseract.so.4.0.0"
  IMPORTED_SONAME_NOCONFIG "libtesseract.so.4.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS libtesseract )
list(APPEND _IMPORT_CHECK_FILES_FOR_libtesseract "${_IMPORT_PREFIX}/lib/libtesseract.so.4.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
