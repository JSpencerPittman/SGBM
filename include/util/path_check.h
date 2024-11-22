#ifndef PATH_H_
#define PATH_H_

#include <stdexcept>
#include <util/format.hpp>
#include <filesystem>

#include<iostream>

#include "util/result.hpp"

namespace fs = std::filesystem;


Status doesPathExist(const fs::path& path) noexcept {
    if (!fs::exists(path))
        return Status(false,
            stringFormat("Provided path does not exists: %s", path.c_str()));
    else return Status(true);
}

Status doesRegularFileExist(const fs::path& path) noexcept {
    Status exists = doesPathExist(path);
    if(!exists.succeed) return exists;
    else if (!fs::is_regular_file(path))
        return Status(false,
            stringFormat("Provided path is not a regular file: %s", path.c_str()));
    else return Status(true);
}

Status doesDirectoryExist(const fs::path& path) noexcept {
    if(path.empty()) return Status(true); // empty path just means current directory
    Status exists = doesPathExist(path);
    if(!exists.succeed) return exists;
    else if (!fs::is_directory(path))
        return Status(false,
            stringFormat("Provided path is not a directory: %s", path.c_str()));
    else return Status(true);
}

Status isValidFilePath(const fs::path& path, std::string requiredExtension = "") {
    if(path.empty())
        return Status(false, "Provided path is empty.");
    else if(!path.has_filename())
        return Status(false,
            stringFormat("Provided path does not have a filename: %s", path.c_str()));
    else if(!path.has_extension() || path.extension() == ".")
        return Status(false,
            stringFormat("Provided path does not have an extension: %s", path.c_str()));
    else if(!requiredExtension.empty() && path.extension() != requiredExtension)
        return Status(false,
            stringFormat("Expected extension %s but instead extension is %s",
                         requiredExtension.c_str(), path.extension().c_str()));
   else return Status(true);
}

Status isValidDirPath(const fs::path& path) {
    if(path.empty())
        return Status(true);
    else if(path.has_extension())
        return Status(false,
            stringFormat("A directory can not have an extension: %s", path.extension().c_str()));
   else return Status(true);
}

#endif