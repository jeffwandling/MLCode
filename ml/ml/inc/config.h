#ifndef _PROG_H_
#define _PROG_H_

#define SAFE_DELETE(x) do { if ((x) != NULL) { delete (x); (x) = NULL;}} while(0);
#define SAFE_DELETEA(x) do { if ((x) != NULL) { delete [] (x); (x) = NULL;}} while(0);

#endif

