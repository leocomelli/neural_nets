# -*- mode: ruby -*-
# vi: set ft=ruby :


VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(2) do |config|
  config.vm.box_download_insecure = true
  config.vm.box = "leocomelli/python"
  config.vm.box_version = "1.0"

  config.vm.synced_folder ".", "/neural_nets", type: "virtualbox"

  config.vm.provision "shell" do |s|
    s.args   = ENV['https_proxy']
    s.inline = <<-SCRIPT
      apt install -y gcc python-scipy python-matplotlib
      pip install --proxy=${1} pybrain
    SCRIPT
  end
end
