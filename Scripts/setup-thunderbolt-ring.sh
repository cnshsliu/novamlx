#!/bin/bash
#
# setup-thunderbolt-ring.sh
#
# Helps set up stable private IPs on the Thunderbolt Bridge between two Macs
# so that MLX Ring transport (and future JACCL) can work reliably.
#
# Usage:
#   ./Scripts/setup-thunderbolt-ring.sh
#
# Recommended subnet: 10.42.0.0/24 (commonly used for direct Thunderbolt links)
#

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  NovaMLX Thunderbolt Ring Setup Helper"
echo "  (for MLX Ring backend over direct Thunderbolt)"
echo "═══════════════════════════════════════════════════════════════"
echo

echo "Detecting Thunderbolt-related interfaces..."
echo

# Show all non-loopback IPv4 addresses
echo "Current IPv4 addresses (excluding 127.0.0.1):"
ifconfig | awk '
/^[a-z0-9]+:/ { iface=$1 }
/inet / { 
    ip=$2
    if (ip != "127.0.0.1") {
        gsub(":", "", iface)
        printf "  %-20s %s\n", iface, ip
    }
}
'

echo
echo "Looking for Thunderbolt Bridge..."
THUNDERBOLT_IFACE=$(networksetup -listallhardwareports 2>/dev/null | awk '
    /Hardware Port: Thunderbolt Bridge/ { getline; print $2 }
' || echo "")

if [ -n "$THUNDERBOLT_IFACE" ]; then
    echo "  Found Thunderbolt Bridge interface: $THUNDERBOLT_IFACE"
    echo "  Current config:"
    networksetup -getinfo "Thunderbolt Bridge" 2>/dev/null || echo "    (could not query via networksetup)"
else
    echo "  Could not auto-detect Thunderbolt Bridge via networksetup."
    echo "  Common names: en5, en6, bridge0, Thunderbolt Bridge"
fi

echo
echo "───────────────────────────────────────────────────────────────"
echo "RECOMMENDED ACTION (do this on BOTH machines):"
echo "───────────────────────────────────────────────────────────────"
echo
echo "1. On each Mac, go to:"
echo "   System Settings → Network → Thunderbolt Bridge → Details..."
echo
echo "2. TCP/IP tab → Configure IPv4 → Manually"
echo
echo "3. Assign stable private IPs on the same /24 subnet, e.g.:"
echo
echo "   Coordinator (M4 Max):"
echo "      IP Address:       10.42.0.1"
echo "      Subnet Mask:      255.255.255.0"
echo "      Router:           (leave blank)"
echo
echo "   Worker (M4 Mac Mini):"
echo "      IP Address:       10.42.0.2"
echo "      Subnet Mask:      255.255.255.0"
echo "      Router:           (leave blank)"
echo
echo "4. Apply / OK, then wait 5–10 seconds for the interface to come up."
echo
echo "5. Verify with:"
echo "   ping 10.42.0.2     (from coordinator)"
echo "   ping 10.42.0.1     (from worker)"
echo
echo "───────────────────────────────────────────────────────────────"
echo "Once both machines have stable IPs, run NovaMLX and it will"
echo "automatically prefer non-link-local addresses for Ring."
echo
echo "If you want to force a specific hostfile, you can also set:"
echo "   ClusterConfig with explicit networkHost values."
echo "───────────────────────────────────────────────────────────────"

# Optional: generate a sample hostfile JSON if user provides IPs
if [ $# -ge 2 ]; then
    COORD_IP=$1
    WORKER_IP=$2
    PORT=8900

    echo
    echo "Sample hostfile JSON for $COORD_IP <-> $WORKER_IP:"
    echo '[
  ["'$COORD_IP':'$PORT'", "'$COORD_IP':'$(($PORT+1))'"],
  ["'$WORKER_IP':'$PORT'", "'$WORKER_IP':'$(($PORT+1))'"]
]'
    echo
fi

echo "Done. After setting static IPs, restart both NovaMLX instances."

# Generate ready-to-use hostfile if both IPs were provided as arguments
if [ $# -ge 2 ]; then
    COORD_IP=$1
    WORKER_IP=$2
    PORT=8900

    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "  READY-TO-USE HOSTFILE (for 10.42.0.1 <-> 10.42.0.2)"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    cat << EOF
[
  ["${COORD_IP}:${PORT}", "${COORD_IP}:$(($PORT+1))"],
  ["${WORKER_IP}:${PORT}", "${WORKER_IP}:$(($PORT+1))"]
]
EOF

    echo
    echo "To enable Ring in code (temporary for testing), change this line in ClusterModelManager.swift:"
    echo
    echo "    if false && engines.count == 2,"
    echo "to"
    echo "    if true && engines.count == 2,"
    echo
    echo "Or better: set enableRingTransport = true in your ClusterConfig."
    echo "═══════════════════════════════════════════════════════════════"
fi

echo
echo "Ring transport can now be enabled via ClusterConfig.enableRingTransport = true"
