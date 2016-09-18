# -*- mode: ruby -*-
# vi: set ft=ruby :


VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(2) do |config|
  config.vm.box_download_insecure = true
  config.vm.box = "leocomelli/python"
  config.vm.box_version = "1.0"

  config.vm.synced_folder ".", "/neural_networks", type: "virtualbox"
end
