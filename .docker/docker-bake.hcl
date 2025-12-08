# docker-bake.hcl for Corral
variable "VERSION" {
}

variable "PYTHON_VERSION" {
}

variable "ORGANIZATION" {
  default = "corral"
}

variable "REGISTRY" {
}

variable "PLATFORMS" {
  default = ["linux/amd64", "linux/arm64"]
}

variable "TARGETS" {
  default = ["corral-envs", "corral-agent-runner"]
}

function "tags" {
  params = [image]
  result = [
    "${REGISTRY}${ORGANIZATION}/${image}"
  ]
}

group "default" {
  targets = "${TARGETS}"
}

target "corral-envs" {
  tags = tags("corral-envs")
  context = "corral-envs"
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
  args = {
    "PYTHON_VERSION" = "${PYTHON_VERSION}"
  }
}

target "corral-agent-runner" {
  tags = tags("corral-agent-runner")
  context = "corral-agent-runner"
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
}
