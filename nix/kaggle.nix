# kaggle 2.2.x — the version omnirun needs (`kaggle>=2.2`) for OAuth
# `credentials.json` support. nixpkgs ships an ancient 1.7.4.5 whose
# `KaggleApi.authenticate()` only understands the legacy `kaggle.json`, so a
# newer wheel is built here. Imported by omnirun, so build it for its python.
{ lib, python3Packages, fetchPypi }:
let
  pp = python3Packages;

  # kaggle 2.2's split-out gRPC/proto client, not in nixpkgs.
  kagglesdk = pp.buildPythonPackage rec {
    pname = "kagglesdk";
    version = "0.1.34";
    format = "wheel";
    src = fetchPypi {
      pname = "kagglesdk";
      inherit version format;
      dist = "py3";
      python = "py3";
      sha256 = "eac37bca83afa20973d0895cc11505f0d502ba46a22a545728ed65e2761a6dcd";
    };
    propagatedBuildInputs = [ pp.protobuf pp.requests ];
    doCheck = false;
    pythonImportsCheck = [ "kagglesdk" ];
  };
in
pp.buildPythonPackage rec {
  pname = "kaggle";
  version = "2.2.3";
  format = "wheel";
  src = fetchPypi {
    pname = "kaggle";
    inherit version format;
    dist = "py3";
    python = "py3";
    sha256 = "535eed6612910979ff3025f9c213207718305d47ab5f263fe74acd735c7e902e";
  };
  propagatedBuildInputs = [
    pp.bleach
    pp.jupytext
    kagglesdk
    pp.packaging
    pp.protobuf
    pp.python-dateutil
    pp.python-dotenv
    pp.python-slugify
    pp.requests
    pp.tqdm
    pp.urllib3
  ];
  doCheck = false;
  # No pythonImportsCheck: `import kaggle` eagerly mkdir's a config dir under
  # $HOME, which the sandbox's read-only /homeless-shelter forbids. The daemon's
  # `backends check` validates kaggle for real at runtime.
  meta = with lib; {
    description = "Official Kaggle API (2.2.x, OAuth credentials.json support)";
    homepage = "https://github.com/Kaggle/kaggle-api";
    license = licenses.asl20;
  };
}
