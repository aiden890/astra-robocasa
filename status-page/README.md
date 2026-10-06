# Status page

Empty main and video pages using the user's existing RoboCasa tracking template:
http://100.86.183.64:8899/

`assets/site.css`, `assets/theme.css`, and `assets/theme.js` are copied from that
page. Content and runtime integrations are intentionally omitted. Serve only
this directory, never the repository root.

The video tab now publishes MP4 sequences of saved model-call observations from
the two 64-step smoke rollouts. These are not continuous recordings. Media and
source observations remain outside Git; catalog.json records their provenance.
