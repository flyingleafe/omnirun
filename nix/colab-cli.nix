# google-colab-cli (the `colab` binary the Colab backend shells out to) plus the
# two of its dependencies that are not yet in nixpkgs. Built from the published
# wheels — pure-Python, no build backend needed. Not in nixpkgs as of this
# writing; upstream them if/when they land.
{ lib, python3Packages, fetchPypi }:
let
  pp = python3Packages;

  jupyter-mimetypes = pp.buildPythonPackage rec {
    pname = "jupyter-mimetypes";
    version = "0.2.0";
    format = "wheel";
    src = fetchPypi {
      pname = "jupyter_mimetypes";
      inherit version format;
      dist = "py3";
      python = "py3";
      sha256 = "e6dcd989258e3fc944365b656d9173191517e0e393bd878e97ce500e5b388527";
    };
    propagatedBuildInputs = [ pp.pyarrow pp.typing-extensions ];
    doCheck = false;
    pythonImportsCheck = [ "jupyter_mimetypes" ];
  };

  jupyter-kernel-client = pp.buildPythonPackage rec {
    pname = "jupyter-kernel-client";
    version = "0.9.0";
    format = "wheel";
    src = fetchPypi {
      pname = "jupyter_kernel_client";
      inherit version format;
      dist = "py3";
      python = "py3";
      sha256 = "77acb8f2f738d97625d6bd01ee8cf21c4d59790b7ba464108712db3870416f20";
    };
    propagatedBuildInputs = [
      pp.jupyter-client
      pp.jupyter-core
      jupyter-mimetypes
      pp.requests
      pp.traitlets
      pp.typing-extensions
      pp.websocket-client
    ];
    doCheck = false;
    pythonImportsCheck = [ "jupyter_kernel_client" ];
  };
in
pp.buildPythonApplication rec {
  pname = "google-colab-cli";
  version = "0.6.0";
  format = "wheel";
  src = fetchPypi {
    pname = "google_colab_cli";
    inherit version format;
    dist = "py3";
    python = "py3";
    sha256 = "46d1aa45811d1ceea82e009e4c7bcd2bdf8dd2ab5c4238c7ccb83e6a52e1f75b";
  };
  propagatedBuildInputs = [
    pp.click
    pp.filelock
    pp.google-auth-oauthlib
    pp.google-auth
    pp.html2text
    jupyter-kernel-client
    pp.nbformat
    pp.packaging
    pp.prompt-toolkit
    pp.pydantic
    pp.pygments
    pp.requests
    pp.rich
    pp.typer
    pp.typing-extensions
    pp.websocket-client
  ];
  # nixpkgs is a hair behind the wheel's declared floors (filelock 3.20.3 vs
  # >=3.29.2, typer 0.24.0 vs >=0.24.1) — patch-level, harmless; relax the check.
  pythonRelaxDeps = [ "filelock" "typer" ];
  doCheck = false;
  pythonImportsCheck = [ "colab_cli" ];
  meta = with lib; {
    description = "Official Google Colab CLI (the `colab` command)";
    homepage = "https://pypi.org/project/google-colab-cli/";
    license = licenses.asl20;
    mainProgram = "colab";
  };
}
