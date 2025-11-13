"""Start-up script adapted from https://github.com/emulab/my-profile"""

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg

# Create a portal context.
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

node = request.RawPC("node")

node.hardware_type = 'c4130'

# Install and execute a script that is contained in the repository.
# node.addService(pg.Execute(shell="sh", command="chmod +x /local/repository/setup.sh"))
node.addService(pg.Execute(shell="sh", command="/local/repository/setup.sh"))

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
