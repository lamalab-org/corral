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
  default = ["corral-envs-base", "corral-agent-runner"]
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

# CI publishing group - images that get pushed to GHCR on main branch
group "ci-publish" {
  targets = ["corral-envs-base", "corral-agent-runner"]
}

# Base image with corral installed but no environment package
# Used as the base for environment-specific images
target "corral-envs-base" {
  tags = tags("corral-envs-base")
  context = "corral-envs"
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
  args = {
    "PYTHON_VERSION" = "${PYTHON_VERSION}"
    # No ENV_PACKAGE_URL - this builds the base image without any environment
  }
}

# Standalone environment image with a specific environment package installed
# Use this target for local testing with a specific environment
target "corral-envs" {
  tags = tags("corral-envs")
  context = "corral-envs"
  contexts = {
    src = ".."
  }
  platforms = "${PLATFORMS}"
  args = {
    "PYTHON_VERSION" = "${PYTHON_VERSION}"
    # Set ENV_PACKAGE_URL to install a specific environment:
    # ENV_PACKAGE_URL = "git+https://github.com/org/my-env.git"
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
