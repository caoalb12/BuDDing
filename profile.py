"""Start-up script adapted from https://github.com/emulab/my-profile"""

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg

# Create a portal context.
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()
 
# Add a raw PC to the request.
pc.defineParameter("cluster", "Cluster", portal.ParameterType.STRING,
                  "wisc.cloudlab.us", ["wisc.cloudlab.us", "clemson.cloudlab.us", "utah.cloudlab.us"])
params = pc.bindParameters()

node = request.RawPC("node")

if params.cluster == "wisc.cloudlab.us":
    node.disk_image = "urn:publicid:IDN+wisc.cloudlab.us+image+emulab-ops:UBUNTU20-64-STD"
elif params.cluster == "clemson.cloudlab.us":
    node.disk_image = "urn:publicid:IDN+clemson.cloudlab.us+image+emulab-ops:UBUNTU20-64-STD"
elif params.cluster == "utah.cloudlab.us":
    node.disk_image = "urn:publicid:IDN+utah.cloudlab.us+image+emulab-ops:UBUNTU20-64-STD"

node.hardware_type = 'c4130'

# Install and execute a script that is contained in the repository.
# node.addService(pg.Execute(shell="sh", command="chmod +x /local/repository/setup.sh"))
node.addService(pg.Execute(shell="sh", command="/local/repository/setup.sh"))

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
