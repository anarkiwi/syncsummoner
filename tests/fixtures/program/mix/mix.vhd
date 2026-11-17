-- SPDX-License-Identifier: Apache-2.0
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.core_pkg.all;
use work.video_stream_pkg.all;

architecture mix of program_top is
  signal s_dry  : unsigned(9 downto 0) := (others => '0');
  signal s_wet  : unsigned(9 downto 0) := (others => '0');
  signal s_out  : unsigned(9 downto 0);
  signal s_dly1 : t_video_stream_yuv444_30b;
  signal s_dly2 : t_video_stream_yuv444_30b;
  signal s_dly3 : t_video_stream_yuv444_30b;
  signal s_dly4 : t_video_stream_yuv444_30b;
begin
  process (clk)
  begin
    if rising_edge(clk) then
      s_dry <= unsigned(data_in.y);
      if registers_in(6)(0) = '1' then
        s_wet <= not unsigned(data_in.y);
      else
        s_wet <= unsigned(data_in.y) + unsigned(registers_in(0)(9 downto 6));
      end if;
      s_dly1 <= data_in;
      s_dly2 <= s_dly1;
      s_dly3 <= s_dly2;
      s_dly4 <= s_dly3;
      data_out   <= s_dly4;
      data_out.y <= std_logic_vector(s_out);
    end if;
  end process;

  blend : entity work.interpolator_u
    generic map (G_WIDTH => 10, G_FRAC_BITS => 10, G_OUTPUT_MIN => 0, G_OUTPUT_MAX => 1023)
    port map (
      clk    => clk,
      enable => '1',
      a      => s_dry,
      b      => s_wet,
      t      => unsigned(registers_in(7)),
      result => s_out,
      valid  => open
    );
end architecture;
