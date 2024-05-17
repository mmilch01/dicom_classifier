#!/bin/bash

# Check if the NB_UID and NB_GID environment variables are set
if [ -n "$NB_UID" ] && [ -n "$NB_GID" ]; then
    # Create jovyan user with NB_UID and NB_GID
    USER_ID=$NB_UID
    GROUP_ID=$NB_GID

    groupadd -o -r jovyan -g ${GROUP_ID}
    useradd --uid ${USER_ID} --gid ${GROUP_ID} -d /home/jovyan -m -s /bin/bash jovyan

    # Change ownership of the /home/jovyan folder to the jovyan user
    chown -R ${USER_ID}:${GROUP_ID} /home/jovyan

    # Start as jovyan user
    exec gosu jovyan "$@"
else
    exec "$@"
fi