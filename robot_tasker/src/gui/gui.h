#ifndef _GUI_H_
#define _GUI_H_

#include "../global_typedef.h"
namespace gui
{

void initiate_gui(const string& global_config);
void gui_render_begin();
void gui_render_finalize();
void gui_render_close();

bool is_gui_opened();

} //namespace {gui}

#endif
