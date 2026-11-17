-- SPDX-License-Identifier: Apache-2.0
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.core_pkg.all;
use work.video_stream_pkg.all;

architecture tiny of program_top is
  signal s_gain : unsigned(3 downto 0)         := (others => '0');
  signal s_on   : std_logic                    := '0';
  signal s_bias : unsigned(3 downto 0)         := (others => '0');
  signal s_far  : std_logic                    := '0';
  signal s_y    : unsigned(9 downto 0)         := (others => '0');
begin
  process (clk)
  begin
    if rising_edge(clk) then
      s_gain <= unsigned(registers_in(0)(9 downto 6));
      s_on   <= registers_in(6)(0);
      s_bias <= unsigned(registers_in(1)(3 downto 0));
      s_far  <= registers_in(1)(9);
      if s_on = '1' then
        s_y <= unsigned(data_in.y) + s_gain;
      elsif s_far = '1' then
        s_y <= unsigned(data_in.y) + s_bias;
      else
        s_y <= unsigned(data_in.y);
      end if;
      data_out   <= data_in;
      data_out.y <= std_logic_vector(s_y);
    end if;
  end process;
end architecture;
