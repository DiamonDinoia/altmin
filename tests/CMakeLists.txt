
# For now we are not using Catch2 

# CPMAddPackage(
#         NAME Catch2
#         GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#         GIT_TAG ${CATCH2_DOWNLOAD_VERSION}
#         EXCLUDE_FROM_ALL YES
#         GIT_SHALLOW YES
# )

# include(CTest)
# enable_testing()

#add_executable(hi test_hello.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)


#install(TARGETS hi DESTINATION ${CMAKE_SOURCE_DIR}/tests)

# CPMAddPackage(
#         NAME altmin
#         GIT_REPOSITORY https://github.com/DiamonDinoia/online-alt-min.git
#         GIT_TAG master
#         EXCLUDE_FROM_ALL YES
#         GIT_SHALLOW YES
#         DOWNLOAD_ONLY YES
# )
# the directory name here altmin can be changed to anything. 
# this might be usefdul if we call this project altmin to void confusion
# also if you only need a subset of the files in the repo, you can specify them here 
# and download/intstall only those files

# Possible change 3

#install(DIRECTORY ${altmin_SOURCE_DIR}/ DESTINATION ${CMAKE_SOURCE_DIR}/artifacts/altmin)
# install(FILES ${CMAKE_SOURCE_DIR}/src/control_flow.py DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)


